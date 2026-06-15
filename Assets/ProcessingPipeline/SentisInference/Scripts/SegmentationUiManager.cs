using System.Collections;
using Meta.XR;
using UnityEngine;
using UnityEngine.UI;

namespace CVMQ3.ProcessingPipeline
{
    public class SegmentationUiManager : MonoBehaviour
    {
        // ── Inspector ─────────────────────────────────────────────────────────
        [Header("Scene references")]
        [SerializeField]
        private EnvironmentRayCastSampleManager m_environmentRaycast;

        [SerializeField]
        private PassthroughCameraAccess m_cameraAccess;

        [SerializeField]
        private SegmentationRunManager m_runManager;

        [SerializeField]
        private HeadGazeSelector m_gazeSelector;

        [Header("UI — same canvas/parent as bounding boxes")]
        [SerializeField]
        private RectTransform m_overlayPrefab;

        [Header("Debug Placeholder")]
        [Tooltip(
            "When SegmentationRunManager.m_debugPlaceholder is on, this sphere is shown at "
                + "the bounding-box centre. Leave unassigned to auto-create a Unity primitive. "
                + "If assigned, keep it INACTIVE in the scene — only the Instantiate clone is shown."
        )]
        [SerializeField]
        private GameObject m_debugSpherePrimitive;

        [Tooltip("World-space radius of the debug placeholder sphere.")]
        [SerializeField]
        private float m_debugSphereRadius = 0.04f;

        // ── Private state ─────────────────────────────────────────────────────
        private RectTransform m_overlayRect;
        private RawImage m_rawImage;

        private Texture2D m_maskTex;
        private Color32[] m_maskPixels;

        private System.Threading.Tasks.Task<Color32[]> m_maskTask;
        private int m_coroutineSerial;

        private GameObject m_placeholderSphere;

        // ── Lifecycle ─────────────────────────────────────────────────────────
        private void Awake()
        {
            if (m_debugSpherePrimitive == null)
                Debug.Log(
                    "[SegUI] No debug sphere primitive assigned — "
                        + "a Unity primitive will be created automatically."
                );
        }

        private void Start()
        {
            m_runManager.OnSegmentationReady += OnSegmentationReady;
            if (m_overlayPrefab != null)
                m_overlayPrefab.gameObject.SetActive(false);
        }

        private void OnEnable()
        {
            if (m_runManager != null)
                m_runManager.OnSegmentationReady += OnSegmentationReady;
            if (m_gazeSelector != null)
                m_gazeSelector.OnSelectionChanged += OnGazeSelectionChanged;
        }

        private void OnDisable()
        {
            if (m_runManager != null)
                m_runManager.OnSegmentationReady -= OnSegmentationReady;
            if (m_gazeSelector != null)
                m_gazeSelector.OnSelectionChanged -= OnGazeSelectionChanged;
        }

        private void OnDestroy()
        {
            if (m_maskTex)
                Destroy(m_maskTex);
            if (m_overlayRect)
                Destroy(m_overlayRect.gameObject);
            if (m_placeholderSphere)
                Destroy(m_placeholderSphere);
        }

        // ── Gaze cleared → hide overlays ─────────────────────────────────────
        private void OnGazeSelectionChanged(MultiDetectionUiManager.BoundingBoxData selected)
        {
            if (selected != null)
                return;
            if (m_placeholderSphere)
                m_placeholderSphere.SetActive(false);
            if (m_overlayRect != null)
                m_overlayRect.gameObject.SetActive(false);
        }

        // ── Entry ─────────────────────────────────────────────────────────────
        private void OnSegmentationReady(SegmentationRunManager.SegResult result, Pose cameraPose)
        {
            if (result.isPlaceholder)
            {
                PlaceholderSphere(result.cropViewport, cameraPose);
                return;
            }

            m_coroutineSerial++;
            int mySerial = m_coroutineSerial;

            // Capture value-type copies for the thread-pool task.
            int[] classMapCopy = result.classMap;
            int mapW = result.mapW;
            int mapH = result.mapH;
            int targetClassId = result.targetClassId;

            m_maskTask = System.Threading.Tasks.Task.Run(() =>
                ComputeMaskPixels(classMapCopy, mapW, mapH, targetClassId)
            );

            StartCoroutine(ApplyMaskWhenReady(result, cameraPose, mySerial));
        }

        // ── Mask application ──────────────────────────────────────────────────
        private IEnumerator ApplyMaskWhenReady(
            SegmentationRunManager.SegResult result,
            Pose cameraPose,
            int mySerial
        )
        {
            while (!m_maskTask.IsCompleted)
                yield return null;

            // Discard if a newer result arrived while we were waiting.
            if (mySerial != m_coroutineSerial)
            {
                Debug.Log("[SegUI] Discarding stale mask (newer pending).");
                yield break;
            }

            if (m_maskTask.IsFaulted)
            {
                Debug.LogError($"[SegUI] Mask task failed: {m_maskTask.Exception}");
                yield break;
            }

            // Resize the mask texture if needed.
            if (
                m_maskTex == null
                || m_maskTex.width != result.mapW
                || m_maskTex.height != result.mapH
            )
            {
                if (m_maskTex != null)
                    Destroy(m_maskTex);
                m_maskTex = new Texture2D(result.mapW, result.mapH, TextureFormat.RGBA32, false);
            }

            // Upload pixels — the task result covers the full mapW × mapH output,
            // already with a Y-flip applied in ComputeMaskPixels.
            m_maskTex.SetPixels32(m_maskTask.Result);
            m_maskTex.Apply();

            PlaceOverlay(m_maskTex, result.cropViewport, result.modelInputUVRect, cameraPose);

            Debug.Log(
                $"[SegUI] Placed overlay '{result.label}' " + $"mask={result.mapW}x{result.mapH}"
            );
        }

        // ── Mask pixel computation (thread-pool safe) ─────────────────────────
        private static Color32[] ComputeMaskPixels(
            int[] classMap,
            int mapW,
            int mapH,
            int targetClassId
        )
        {
            Color32 tint = HsvToRgb32((Mathf.Abs(targetClassId * 0.137f) % 1f), 0.9f, 1f, 0.55f);
            var clear = new Color32(0, 0, 0, 0);
            var pixels = new Color32[mapH * mapW];

            // Y-flip: model output row 0 is the top of the image; Texture2D row 0
            // is the bottom.  Flip vertically so the overlay is the right way up.
            for (int i = 0; i < classMap.Length; i++)
            {
                int row = i / mapW;
                int col = i % mapW;
                int flippedRow = (mapH - 1 - row);
                pixels[flippedRow * mapW + col] = classMap[i] == targetClassId ? tint : clear;
            }

            return pixels;
        }

        private static Color32 HsvToRgb32(float h, float s, float v, float a)
        {
            float r,
                g,
                b;
            if (s == 0f)
            {
                r = g = b = v;
            }
            else
            {
                h *= 6f;
                int i = (int)h;
                float f = h - i;
                float p = v * (1f - s);
                float q = v * (1f - s * f);
                float t = v * (1f - s * (1f - f));
                switch (i % 6)
                {
                    case 0:
                        r = v;
                        g = t;
                        b = p;
                        break;
                    case 1:
                        r = q;
                        g = v;
                        b = p;
                        break;
                    case 2:
                        r = p;
                        g = v;
                        b = t;
                        break;
                    case 3:
                        r = p;
                        g = q;
                        b = v;
                        break;
                    case 4:
                        r = t;
                        g = p;
                        b = v;
                        break;
                    default:
                        r = v;
                        g = p;
                        b = q;
                        break;
                }
            }
            return new Color32((byte)(r * 255), (byte)(g * 255), (byte)(b * 255), (byte)(a * 255));
        }

        // ── Overlay placement ─────────────────────────────────────────────────
        private void PlaceOverlay(Texture2D tex, Vector4 cropViewport, Vector4 modelInputUVRect, Pose cameraPose)
        {
            float vx1 = cropViewport.x,
                vy1 = cropViewport.y;
            float vx2 = cropViewport.z,
                vy2 = cropViewport.w;

            var normCenter = new Vector2((vx1 + vx2) * 0.5f, (vy1 + vy2) * 0.5f);
            var centerRay = m_cameraAccess.ViewportPointToRay(normCenter, cameraPose);

            var worldHit = m_environmentRaycast.Raycast(centerRay);
            if (!worldHit.HasValue)
            {
                Debug.LogWarning("[SegUI] PlaceOverlay: environment raycast missed.");
                return;
            }

            float distance = Vector3.Distance(cameraPose.position, worldHit.Value);
            var worldCenter = centerRay.GetPoint(distance);
            var normal = (worldCenter - cameraPose.position).normalized;
            var plane = new Plane(normal, worldCenter);

            var minRay = m_cameraAccess.ViewportPointToRay(new Vector2(vx1, vy1), cameraPose);
            var maxRay = m_cameraAccess.ViewportPointToRay(new Vector2(vx2, vy2), cameraPose);
            plane.Raycast(minRay, out float dMin);
            plane.Raycast(maxRay, out float dMax);

            var invRot = Quaternion.Inverse(cameraPose.rotation);
            var minLocal = invRot * (minRay.GetPoint(dMin) - cameraPose.position);
            var maxLocal = invRot * (maxRay.GetPoint(dMax) - cameraPose.position);
            var worldSize = new Vector2(
                Mathf.Abs(maxLocal.x - minLocal.x),
                Mathf.Abs(maxLocal.y - minLocal.y)
            );

            EnsureOverlayRect();
            m_rawImage.texture = tex;
            m_rawImage.uvRect = new Rect(
                modelInputUVRect.x,
                modelInputUVRect.y,
                modelInputUVRect.z - modelInputUVRect.x,
                modelInputUVRect.w - modelInputUVRect.y
            );

            m_overlayRect.SetPositionAndRotation(worldCenter, Quaternion.LookRotation(normal));
            m_overlayRect.sizeDelta = worldSize;
            m_overlayRect.gameObject.SetActive(true);

            Debug.Log(
                $"[SegUI] PlaceOverlay viewport=({vx1:F3},{vy1:F3},{vx2:F3},{vy2:F3}) "
                    + $"worldCenter={worldCenter:F3} worldSize={worldSize}"
            );
        }

        private void EnsureOverlayRect()
        {
            if (m_overlayRect != null)
                return;
            if (m_overlayPrefab == null)
            {
                Debug.LogError("[SegUI] m_overlayPrefab not assigned!");
                return;
            }
            Transform parent = m_overlayPrefab.parent;
            m_overlayRect = Instantiate(m_overlayPrefab, parent);
            m_rawImage = m_overlayRect.GetComponent<RawImage>();
            if (m_rawImage == null)
                Debug.LogError("[SegUI] Overlay prefab has no RawImage component!");
            m_overlayRect.gameObject.SetActive(false);
        }

        // ── Placeholder sphere ────────────────────────────────────────────────
        private void PlaceholderSphere(Vector4 cropViewport, Pose cameraPose)
        {
            float vx1 = cropViewport.x,
                vy1 = cropViewport.y;
            float vx2 = cropViewport.z,
                vy2 = cropViewport.w;

            var normCenter = new Vector2((vx1 + vx2) * 0.5f, (vy1 + vy2) * 0.5f);

            Debug.Log(
                $"[SegUI:Sphere] cropViewport=({vx1:F3},{vy1:F3},{vx2:F3},{vy2:F3}) "
                    + $"normCenter=({normCenter.x:F3},{normCenter.y:F3})"
            );

            var centerRay = m_cameraAccess.ViewportPointToRay(normCenter, cameraPose);
            var worldHit = m_environmentRaycast.Raycast(centerRay);

            if (!worldHit.HasValue)
            {
                Debug.LogWarning(
                    $"[SegUI:Sphere] Raycast MISSED — normCenter=({normCenter.x:F3},{normCenter.y:F3}). "
                        + "Check viewport point is in [0,1]² and environment mesh is present."
                );
                return;
            }

            float distance = Vector3.Distance(cameraPose.position, worldHit.Value);
            var worldCenter = centerRay.GetPoint(distance);

            Debug.Log(
                $"[SegUI:Sphere] hit={worldHit.Value:F3} dist={distance:F3} "
                    + $"worldCenter={worldCenter:F3}"
            );

            EnsurePlaceholderSphere();
            if (!m_placeholderSphere)
            {
                Debug.LogError("[SegUI:Sphere] Sphere is null after EnsurePlaceholderSphere!");
                return;
            }

            var selectedBox = m_gazeSelector?.CurrentSelection;
            var rt = selectedBox?.BoxRectTransform;

            m_placeholderSphere.transform.position = worldCenter;

            if (rt != null)
            {
                float worldWidth = rt.sizeDelta.x * rt.lossyScale.x;
                float worldHeight = rt.sizeDelta.y * rt.lossyScale.y;
                m_placeholderSphere.transform.localScale = new Vector3(
                    worldWidth * 0.5f,
                    worldHeight * 0.25f,
                    worldWidth * 0.5f
                );
            }
            else
            {
                float d = m_debugSphereRadius * 2f;
                m_placeholderSphere.transform.localScale = Vector3.one * d;
            }

            m_placeholderSphere.SetActive(true);

            Debug.Log(
                $"[SegUI:Sphere] Placed at {worldCenter:F3} "
                    + $"scale={m_placeholderSphere.transform.localScale}"
            );
        }

        private void EnsurePlaceholderSphere()
        {
            if (m_placeholderSphere)
            {
                Debug.Log($"[SegUI:Sphere] Reusing '{m_placeholderSphere.name}'.");
                return;
            }

            Debug.Log("[SegUI:Sphere] Creating new placeholder sphere.");

            if (m_debugSpherePrimitive != null)
            {
                m_placeholderSphere = Instantiate(m_debugSpherePrimitive);
                Debug.Log("[SegUI:Sphere] Cloned from m_debugSpherePrimitive.");
            }
            else
            {
                m_placeholderSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);

                var col = m_placeholderSphere.GetComponent<Collider>();
                if (col != null)
                    Destroy(col);

                var rend = m_placeholderSphere.GetComponent<Renderer>();
                if (rend != null)
                {
                    Shader standardShader = Shader.Find("Standard");
                    var mat =
                        standardShader != null
                            ? new Material(standardShader)
                            : new Material(rend.sharedMaterial);
                    mat.color = new Color(1f, 0f, 1f, 0.75f);
                    mat.SetFloat("_Mode", 3f);
                    mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                    mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                    mat.SetInt("_ZWrite", 0);
                    mat.DisableKeyword("_ALPHATEST_ON");
                    mat.EnableKeyword("_ALPHABLEND_ON");
                    mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
                    mat.renderQueue = 3000;
                    rend.material = mat;
                }
            }

            m_placeholderSphere.name = "[SegUI] DebugPlaceholder";
            m_placeholderSphere.SetActive(false);
        }
    }
}
