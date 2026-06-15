using System.Collections;
using Meta.XR;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace CVMQ3.ProcessingPipeline
{
    /// <summary>
    /// Passed from GazeCropExtractor to SegmentationRunManager.
    /// The RT is model-ready.
    /// </summary>
    public struct CropInfo
    {
        /// <summary>
        /// Normalised viewport rect of the original bbox crop in full-camera
        /// space (0-1, Y-up).  Components: x1, y1, x2, y2.
        /// Used by SegmentationUiManager to place the overlay back in the scene.
        /// </summary>
        public Vector4 ViewportNorm;

        /// <summary>
        /// UV rect inside the 640×640 model input that contains real image pixels
        /// (i.e. the non-padded region, in [0,1] texture UV space, Y-up).
        /// x = uMin, y = vMin, z = uMax, w = vMax.
        /// </summary>
        public Vector4 ModelInputUVRect;
    }

    /// <summary>
    /// Extracts the gaze-selected bounding box from the passthrough camera
    /// texture, resizes/letterboxes it to the model's input resolution, and
    /// fires <see cref="OnCropReady"/> with the resulting RenderTexture.
    ///
    /// Resize policy (two sequential GPU Blits):
    ///   • Crop larger  than modelInputSize : scale DOWN to fit, then pad.
    ///   • Crop smaller than modelInputSize : pad directly (no upscaling).
    /// Output RT is always modelInputSize × modelInputSize with black bars.
    /// </summary>
    public class GazeCropExtractor : MonoBehaviour
    {
        // ── Inspector ─────────────────────────────────────────────────────────
        [Header("References")]
        [SerializeField]
        internal HeadGazeSelector m_gazeSelector;

        [SerializeField]
        private PassthroughCameraAccess m_cameraAccess;

        [Header("Crop Settings")]
        [SerializeField, Range(0f, 0.5f)]
        private float m_padding = 0.1f;

        [SerializeField]
        internal float m_extractInterval = 0.1f;

        [Header("Model Input Size")]
        [Tooltip("Target width AND height sent to the segmentation model (assumed square).")]
        [SerializeField]
        private int m_modelInputSize = 640;

        [Header("Debug — Crop Saving")]
        [Tooltip(
            "When enabled, every extracted crop is saved to persistentDataPath as a PNG. "
                + "Disable for release builds."
        )]
        [SerializeField]
        private bool m_saveCrops = false;

        [Tooltip("Subfolder inside Application.persistentDataPath where crops are written.")]
        [SerializeField]
        private string m_saveSubfolder = "CropSamples";

        // ── Public API ────────────────────────────────────────────────────────

        /// <summary>
        /// Most recently produced model-input RenderTexture.
        /// Always <see cref="m_modelInputSize"/> × <see cref="m_modelInputSize"/>.
        /// </summary>
        public RenderTexture LatestModelInputRT { get; private set; }

        /// <summary>Camera pose captured at the moment of the last extraction.</summary>
        public Pose CachedCameraPose { get; private set; }

        /// <summary>
        /// Fired when a new model-input RT is ready.
        /// The RT is always modelInputSize × modelInputSize.
        /// </summary>
        public System.Action<RenderTexture, CropInfo> OnCropReady;

        // ── Private state ─────────────────────────────────────────────────────
        private float m_timeSinceLastExtract;
        private bool m_extracting;
        private int m_savedCropCount;

        private RenderTexture m_rawCropRT; // variable-size raw bbox crop
        private RenderTexture m_modelInputRT; // fixed modelInputSize² output
        private RenderTexture m_frameSnapshotRT; // full-frame snapshot (debug only)

        // ── Lifecycle ─────────────────────────────────────────────────────────
        private void OnDestroy()
        {
            ReleaseRT(ref m_rawCropRT);
            ReleaseRT(ref m_modelInputRT);
            ReleaseRT(ref m_frameSnapshotRT);
        }

        // ── Update ────────────────────────────────────────────────────────────
        private void Update()
        {
            if (m_gazeSelector == null || m_cameraAccess == null)
                return;
            if (m_gazeSelector.CurrentSelection == null)
                return;
            if (m_extracting)
                return;

            m_timeSinceLastExtract += Time.deltaTime;
            if (m_timeSinceLastExtract < m_extractInterval)
                return;

            m_timeSinceLastExtract = 0f;
            StartCoroutine(ExtractCrop(m_gazeSelector.CurrentSelection));
        }

        // ── Extraction coroutine ──────────────────────────────────────────────
        private IEnumerator ExtractCrop(MultiDetectionUiManager.BoundingBoxData box)
        {
            m_extracting = true;

            if (!m_cameraAccess.IsPlaying)
            {
                Debug.LogWarning("[GazeCrop] Camera not playing.");
                m_extracting = false;
                yield break;
            }

            Texture rawTex = m_cameraAccess.GetTexture();
            if (rawTex == null)
            {
                Debug.LogWarning("[GazeCrop] GetTexture() returned null.");
                m_extracting = false;
                yield break;
            }

            CachedCameraPose = m_cameraAccess.GetCameraPose();

            // ── Padded normalised bbox (Y-down, matching NormalizedBBox convention) ──
            Vector4 bbox = box.NormalizedBBox; // x1,y1,x2,y2 normalised Y-down
            float u1 = bbox.x,
                v1 = bbox.y,
                u2 = bbox.z,
                v2 = bbox.w;

            if (u2 - u1 <= 0f || v2 - v1 <= 0f)
            {
                Debug.LogWarning($"[GazeCrop] Invalid NormalizedBBox on '{box.ClassName}'.");
                m_extracting = false;
                yield break;
            }

            float uPad = (u2 - u1) * m_padding;
            float vPad = (v2 - v1) * m_padding;
            u1 = Mathf.Clamp01(u1 - uPad);
            u2 = Mathf.Clamp01(u2 + uPad);
            v1 = Mathf.Clamp01(v1 - vPad);
            v2 = Mathf.Clamp01(v2 + vPad);

            // ── Pixel coords in source texture (GPU Y origin = bottom-left) ──
            int texW = rawTex.width;
            int texH = rawTex.height;

            int px = Mathf.RoundToInt(u1 * texW);
            int py = Mathf.RoundToInt((1f - v2) * texH); // flip V for GPU origin
            int pw = Mathf.Max(1, Mathf.RoundToInt((u2 - u1) * texW));
            int ph = Mathf.Max(1, Mathf.RoundToInt((v2 - v1) * texH));

            px = Mathf.Clamp(px, 0, texW - 1);
            py = Mathf.Clamp(py, 0, texH - 1);
            pw = Mathf.Clamp(pw, 1, texW - px);
            ph = Mathf.Clamp(ph, 1, texH - py);

            // ── Step 1: GPU blit raw bbox crop (pw × ph) ──────────────────────
            EnsureRT(ref m_rawCropRT, pw, ph);
            Graphics.Blit(
                rawTex,
                m_rawCropRT,
                new Vector2((float)pw / texW, (float)ph / texH),
                new Vector2((float)px / texW, (float)py / texH)
            );

            // ── Step 2: Resize + letterbox into modelInputSize × modelInputSize ─
            //
            // Policy:
            //   lbScale = min(target/pw, target/ph, 1f)
            //     → < 1  when crop is larger than target  (scale down)
            //     → = 1  when crop is smaller than target (no upscale, just pad)
            int target = m_modelInputSize;
            float lbScale = Mathf.Min(1f, Mathf.Min((float)target / pw, (float)target / ph));
            int scaledW = Mathf.Max(1, Mathf.RoundToInt(pw * lbScale));
            int scaledH = Mathf.Max(1, Mathf.RoundToInt(ph * lbScale));
            float padX = (target - scaledW) * 0.5f; // pixels on each horizontal side
            float padY = (target - scaledH) * 0.5f; // pixels on each vertical side

            // UV rect of the real-image region inside the modelInputSize² texture (Y-up).
            // padX/padY are in destination pixels; target = m_modelInputSize.
            float uvXmin = padX / target;
            float uvYmin = padY / target;
            float uvXmax = (padX + scaledW) / target;
            float uvYmax = (padY + scaledH) / target;

            EnsureRT(ref m_modelInputRT, target, target);

            // Clear to black so letterbox bars / pad regions are black.
            var prevActive = RenderTexture.active;
            RenderTexture.active = m_modelInputRT;
            GL.Clear(false, true, Color.black);
            RenderTexture.active = prevActive;

            // Graphics.Blit applies:  dst_uv = src_uv * blitScale + blitOffset
            // We need the source image to occupy the destination sub-rect
            //   x ∈ [padX/target .. (padX+scaledW)/target],
            //   y ∈ [padY/target .. (padY+scaledH)/target]
            // Solving for blitScale/blitOffset (src_uv ∈ [0,1] → dst_uv in sub-rect):
            //   dst_uv = src_uv * (scaledW/target) + (padX/target)
            // So:
            //   blitScale  = scaledW / target
            //   blitOffset = padX    / target
            //
            // BUT Graphics.Blit maps source UVs, not destination UVs, so the
            // transformation is applied in the *source* domain:
            //   src_uv = dst_uv * blitScale + blitOffset
            // To place the image in a centred sub-rect of the destination:
            //   blitScale  = target / scaledW   (source spans more of [0,1] → smaller in dst)
            //   blitOffset = -padX  / scaledW
            float blitScaleX = (float)target / scaledW;
            float blitScaleY = (float)target / scaledH;
            float blitOffsetX = -padX / scaledW;
            float blitOffsetY = -padY / scaledH;

            Graphics.Blit(
                m_rawCropRT,
                m_modelInputRT,
                new Vector2(blitScaleX, blitScaleY),
                new Vector2(blitOffsetX, blitOffsetY)
            );

            LatestModelInputRT = m_modelInputRT;

            // Build CropInfo: viewport rect in Y-up camera space for the UI.
            var info = new CropInfo
            {
                ViewportNorm = new Vector4(u1, 1f - v2, u2, 1f - v1),
                ModelInputUVRect = new Vector4(uvXmin, uvYmin, uvXmax, uvYmax),
            };

            Debug.Log(
                $"[GazeCrop] '{box.ClassName}' raw={pw}x{ph} lbScale={lbScale:F3} "
                    + $"scaled={scaledW}x{scaledH} pad=({padX:F1},{padY:F1}) → {target}x{target}"
            );

            OnCropReady?.Invoke(m_modelInputRT, info);

            // ── Optional debug saves (non-blocking) ───────────────────────────
            if (m_saveCrops)
            {
                if (
                    m_frameSnapshotRT == null
                    || m_frameSnapshotRT.width != texW
                    || m_frameSnapshotRT.height != texH
                )
                {
                    ReleaseRT(ref m_frameSnapshotRT);
                    m_frameSnapshotRT = new RenderTexture(
                        texW,
                        texH,
                        0,
                        RenderTextureFormat.ARGB32
                    );
                    m_frameSnapshotRT.Create();
                }
                Graphics.Blit(rawTex, m_frameSnapshotRT);

                int idx = m_savedCropCount;
                string safeName = SanitiseName(box.ClassName);
                StartCoroutine(SaveRTAsync(m_frameSnapshotRT, $"{idx:D4}_{safeName}_frame"));
                StartCoroutine(SaveRTAsync(m_rawCropRT, $"{idx:D4}_{safeName}_rawcrop"));
                StartCoroutine(SaveRTAsync(m_modelInputRT, $"{idx:D4}_{safeName}_modelinput"));
                m_savedCropCount++;
            }

            m_extracting = false;
        }

        // ── RT helpers ────────────────────────────────────────────────────────

        private static void EnsureRT(ref RenderTexture rt, int w, int h)
        {
            if (rt != null && rt.width == w && rt.height == h)
                return;
            ReleaseRT(ref rt);
            rt = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32);
            rt.Create();
        }

        private static void ReleaseRT(ref RenderTexture rt)
        {
            if (rt == null)
                return;
            rt.Release();
            Destroy(rt);
            rt = null;
        }

        // ── Debug save coroutine ──────────────────────────────────────────────
        private IEnumerator SaveRTAsync(RenderTexture rt, string stem)
        {
            if (rt == null)
                yield break;

            Texture2D snap = null;

            if (SystemInfo.supportsAsyncGPUReadback)
            {
                var req = AsyncGPUReadback.Request(rt, 0, GraphicsFormat.R8G8B8A8_UNorm);
                while (!req.done)
                    yield return null;

                if (req.hasError)
                {
                    Debug.LogWarning($"[GazeCrop] AsyncGPUReadback failed for '{stem}'.");
                    yield break;
                }

                snap = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
                snap.LoadRawTextureData(req.GetData<byte>());
                snap.Apply(false);
            }
            else
            {
                var prevActive = RenderTexture.active;
                RenderTexture.active = rt;
                snap = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
                snap.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0, false);
                snap.Apply(false);
                RenderTexture.active = prevActive;
            }

            byte[] png = snap.EncodeToPNG();
            Destroy(snap);

            if (png == null || png.Length == 0)
            {
                Debug.LogWarning($"[GazeCrop] PNG encode empty for '{stem}'.");
                yield break;
            }

            string folder = System.IO.Path.Combine(Application.persistentDataPath, m_saveSubfolder);
            System.IO.Directory.CreateDirectory(folder);
            string path = System.IO.Path.Combine(
                folder,
                $"{stem}_{System.DateTime.Now:HHmmss_fff}.png"
            );
            System.IO.File.WriteAllBytes(path, png);
            Debug.Log($"[GazeCrop] Saved → {path}");
        }

        private static string SanitiseName(string name) =>
            string.IsNullOrWhiteSpace(name) ? "unknown" : name.Replace(" ", "_").Replace("/", "-");
    }
}
