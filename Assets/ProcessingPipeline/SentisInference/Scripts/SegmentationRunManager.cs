using System.Collections;
using Unity.InferenceEngine;
using UnityEngine;

namespace CVMQ3.ProcessingPipeline
{
    /// <summary>
    /// Runs either:
    ///   • YOLO26n-sem  (semantic)  — single output (1,C,H,W) logits; argmax → classMap.
    ///   • YOLO26n-seg  (instance)  — two outputs: detections (1,4+C+32,8400) and
    ///                                prototype masks (1,32,160,160); decode → classMap.
    ///
    /// Both paths produce the same SegResult / classMap interface so
    /// SegmentationUiManager needs no changes.
    ///
    /// Toggle between modes with <see cref="m_useInstanceSegmentation"/>.
    /// Remember to assign the correct model asset for the chosen mode.
    /// </summary>
    public class SegmentationRunManager : MonoBehaviour
    {
        // ── Inspector ─────────────────────────────────────────────────────────
        [Header("Model")]
        [SerializeField]
        public ModelAsset m_segModel;

        [SerializeField]
        private TextAsset m_segLabels;

        [SerializeField]
        private BackendType m_backend = BackendType.GPUCompute;

        [Header("Mode")]
        [Tooltip(
            "When OFF  → semantic segmentation (YOLO26n-sem): single (1,C,H,W) logits output.\n"
                + "When ON   → instance segmentation (YOLO26n-seg): two outputs — "
                + "detections (1,4+C+32,N) and proto masks (1,32,H,W)."
        )]
        [SerializeField]
        private bool m_useInstanceSegmentation = false;

        [Header("Instance Segmentation Thresholds")]
        [Tooltip("Minimum objectness×class score to consider a detection.")]
        [SerializeField, Range(0f, 1f)]
        internal float m_scoreThreshold = 0.25f;

        [Tooltip("IoU threshold for NMS.")]
        [SerializeField, Range(0f, 1f)]
        internal float m_iouThreshold = 0.45f;

        [Tooltip("Sigmoid threshold to binarise the decoded instance mask (0.5 is standard).")]
        [SerializeField, Range(0f, 1f)]
        private float m_maskThreshold = 0.5f;

        [Header("Debug")]
        [Tooltip(
            "Skip real inference and immediately fire OnSegmentationReady with a placeholder "
                + "result. SegmentationUiManager will show a debug sphere at the bounding-box "
                + "centre so you can verify world-space placement without a working model."
        )]
        [SerializeField]
        private bool m_debugPlaceholder = false;

        [Header("References")]
        [SerializeField]
        private GazeCropExtractor m_cropExtractor;

        [SerializeField]
        private SegmentationUiManager m_uiManager;

        [Header("(Editor only) — Convert ONNX to Sentis")]
        public ModelAsset OnnxModel;

        // ── Result type ───────────────────────────────────────────────────────

        public struct SegResult
        {
            /// <summary>
            /// Per-pixel argmax class ID, row-major, size mapH × mapW.
            /// For instance mode this is a binary mask (0 = background,
            /// targetClassId = foreground) at proto-mask resolution.
            /// Null when isPlaceholder is true.
            /// </summary>
            public int[] classMap;
            public int mapW;
            public int mapH;
            public int targetClassId;
            public string label;

            /// <summary>
            /// Normalised crop region in full-camera viewport space (x1,y1,x2,y2, Y-up).
            /// Passed through unchanged from CropInfo.ViewportNorm.
            /// </summary>
            public Vector4 cropViewport;

            /// <summary>When true the result is a debug placeholder; classMap is null.</summary>
            public bool isPlaceholder;

            public Vector4 modelInputUVRect; // mirrors CropInfo.ModelInputUVRect
        }

        public System.Action<SegResult, Pose> OnSegmentationReady;

        // ── Private state ─────────────────────────────────────────────────────
        private int m_inputW;
        private int m_inputH;
        private int m_numClasses;
        private bool m_dimensionsDerived;

        private Worker m_worker;
        private string[] m_labelNames;
        private Tensor<float> m_inputTensor;

        private bool m_inferenceRunning;
        private float m_inferenceStartTime;
        private float m_lastInferenceTime;

        // NMS working lists — pre-allocated, cleared each call.
        private readonly System.Collections.Generic.List<int> m_nmsFiltered =
            new System.Collections.Generic.List<int>();
        private bool[] m_nmsSuppressed = new bool[0];

        private const float InferenceTimeout = 5f;
        private const float MinInferenceInterval = 0.15f;

        // Number of mask coefficients produced by the instance seg head.
        // Standard for all Ultralytics -seg models (YOLOv8/11/26 etc.).
        private const int MaskCoeffs = 32;

        public int NumClasses => m_numClasses;

        // ── Lifecycle ─────────────────────────────────────────────────────────
        private void Awake()
        {
            if (m_segLabels != null)
            {
                m_labelNames = System.Array.FindAll(
                    m_segLabels.text.Split('\n'),
                    s => !string.IsNullOrWhiteSpace(s)
                );
            }

            if (m_debugPlaceholder)
            {
                Debug.Log("[Seg] Debug placeholder mode — model not loaded.");
                m_numClasses = m_labelNames?.Length ?? 19;
                return;
            }

            if (m_segModel == null)
            {
                Debug.LogError("[Seg] m_segModel is not assigned!");
                return;
            }

            var model = ModelLoader.Load(m_segModel);
            var inputShape = model.inputs[0].shape;

            m_inputH = inputShape.Get(2);
            m_inputW = inputShape.Get(3);
            m_numClasses = m_labelNames?.Length ?? 19;

            m_worker = new Worker(model, m_backend);
            m_inputTensor = new Tensor<float>(new TensorShape(1, 3, m_inputH, m_inputW));

            Debug.Log(
                $"[Seg] Loaded model — mode={(m_useInstanceSegmentation ? "instance" : "semantic")} "
                    + $"input={m_inputW}x{m_inputH} classes={m_numClasses}"
            );
        }

        private void Start()
        {
            m_cropExtractor.OnCropReady += OnCropReady;
        }

        private void Update()
        {
            if (m_inferenceRunning && Time.time - m_inferenceStartTime > InferenceTimeout)
            {
                Debug.LogWarning("[Seg] Inference timeout — resetting flag.");
                m_inferenceRunning = false;
            }
        }

        private void OnDestroy()
        {
            if (m_cropExtractor != null)
                m_cropExtractor.OnCropReady -= OnCropReady;

            m_worker?.Dispose();
            m_inputTensor?.Dispose();
        }

        // ── Entry point ───────────────────────────────────────────────────────
        private void OnCropReady(RenderTexture modelInputRT, CropInfo cropInfo)
        {
            if (m_inferenceRunning)
            {
                Debug.Log("[Seg] Busy — skipping frame.");
                return;
            }

            if (Time.time - m_lastInferenceTime < MinInferenceInterval)
                return;

            if (m_cropExtractor.m_gazeSelector.CurrentSelection == null)
                return;

            if (modelInputRT == null)
            {
                Debug.LogWarning("[Seg] modelInputRT is null.");
                return;
            }

            m_inferenceRunning = true;
            m_inferenceStartTime = Time.time;
            m_lastInferenceTime = Time.time;

            int selectedClassId = m_cropExtractor.m_gazeSelector.CurrentSelection.ClassId;

            // ── Debug placeholder ─────────────────────────────────────────────
            if (m_debugPlaceholder)
            {
                var placeholder = new SegResult
                {
                    classMap = null,
                    mapW = 0,
                    mapH = 0,
                    targetClassId = selectedClassId,
                    label = GetLabel(selectedClassId),
                    cropViewport = cropInfo.ViewportNorm,
                    isPlaceholder = true,
                };
                OnSegmentationReady?.Invoke(placeholder, m_cropExtractor.CachedCameraPose);
                StartCoroutine(ResetInferenceFlagNextFrame());
                return;
            }

            // ── Dispatch to the correct inference coroutine ───────────────────
            if (m_useInstanceSegmentation)
                StartCoroutine(
                    RunSafe(RunInstanceSegmentation(modelInputRT, cropInfo, selectedClassId))
                );
            else
                StartCoroutine(
                    RunSafe(RunSemanticSegmentation(modelInputRT, cropInfo, selectedClassId))
                );
        }

        // ── Coroutine exception wrapper ───────────────────────────────────────
        private IEnumerator RunSafe(IEnumerator inner)
        {
            while (true)
            {
                bool hasNext;
                try
                {
                    hasNext = inner.MoveNext();
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"[Seg] Crashed: {e}");
                    m_inferenceRunning = false;
                    yield break;
                }
                if (!hasNext)
                    break;
                yield return inner.Current;
            }
            m_inferenceRunning = false;
        }

        private IEnumerator ResetInferenceFlagNextFrame()
        {
            yield return null;
            m_inferenceRunning = false;
        }

        // ════════════════════════════════════════════════════════════════════
        //  SEMANTIC SEGMENTATION  (YOLO26n-sem)
        //  Single output: (1, C, mapH, mapW) — argmax over C → classMap
        // ════════════════════════════════════════════════════════════════════
        private IEnumerator RunSemanticSegmentation(
            RenderTexture modelInputRT,
            CropInfo cropInfo,
            int selectedClassId
        )
        {
            Debug.Log($"[Seg/Semantic] Start — targetClass={selectedClassId}");

            var texTransform = new TextureTransform().SetDimensions(m_inputW, m_inputH, 3);
            TextureConverter.ToTensor(modelInputRT, m_inputTensor, texTransform);
            m_worker.Schedule(m_inputTensor);
            yield return null;

            var logitsAwaiter = (m_worker.PeekOutput(0) as Tensor<float>)
                .ReadbackAndCloneAsync()
                .GetAwaiter();
            while (!logitsAwaiter.IsCompleted)
                yield return null;

            using var logits = logitsAwaiter.GetResult();

            if (!m_dimensionsDerived)
            {
                m_dimensionsDerived = true;
                int derived = logits.shape[1];
                if (derived != m_numClasses)
                {
                    Debug.LogWarning($"[Seg] Class count corrected: {m_numClasses} → {derived}");
                    m_numClasses = derived;
                }
                Debug.Log(
                    $"[Seg/Semantic] Output: ({logits.shape[0]},{logits.shape[1]},"
                        + $"{logits.shape[2]},{logits.shape[3]})"
                );
            }

            int mapH = logits.shape[2];
            int mapW = logits.shape[3];

            // Argmax over the class dimension.
            var classMap = new int[mapH * mapW];
            var maxLogits = new float[mapH * mapW];
            for (int i = 0; i < maxLogits.Length; i++)
                maxLogits[i] = float.MinValue;

            for (int c = 0; c < m_numClasses; c++)
            for (int row = 0; row < mapH; row++)
            for (int col = 0; col < mapW; col++)
            {
                int idx = row * mapW + col;
                float v = logits[0, c, row, col];
                if (v > maxLogits[idx])
                {
                    maxLogits[idx] = v;
                    classMap[idx] = c;
                }
            }

            // Guard: target class must be present.
            bool found = false;
            for (int i = 0; i < classMap.Length; i++)
                if (classMap[i] == selectedClassId)
                {
                    found = true;
                    break;
                }

            if (!found)
            {
                Debug.Log(
                    $"[Seg/Semantic] Class {selectedClassId} "
                        + $"('{GetLabel(selectedClassId)}') not in output — skipping."
                );
                yield break;
            }

            FireResult(classMap, mapW, mapH, selectedClassId, cropInfo);
        }

        // ════════════════════════════════════════════════════════════════════
        //  INSTANCE SEGMENTATION  (YOLO26n-seg)
        //
        //  Output 0: (1, 4 + numClasses + 32, 8400)
        //            [cx, cy, w, h, cls_0..cls_C-1, coeff_0..coeff_31]
        //  Output 1: (1, 32, protoH, protoW)   — prototype masks
        //
        //  Pipeline:
        //    1. Score-filter + NMS on output 0, keeping only the best
        //       detection whose class == selectedClassId.
        //    2. matmul(coefficients[1×32], protos[32×(protoH*protoW)])
        //       → raw mask [1×(protoH*protoW)]
        //    3. Sigmoid + threshold at m_maskThreshold.
        //    4. Crop to the detection's bounding box projected onto proto space.
        //    5. Write a classMap where foreground pixels = selectedClassId,
        //       background = 0 (background class or any neutral value).
        // ════════════════════════════════════════════════════════════════════
        private IEnumerator RunInstanceSegmentation(
            RenderTexture modelInputRT,
            CropInfo cropInfo,
            int selectedClassId
        )
        {
            Debug.Log($"[Seg/Instance] Start - targetClass={selectedClassId}");

            var texTransform = new TextureTransform().SetDimensions(m_inputW, m_inputH, 3);
            TextureConverter.ToTensor(modelInputRT, m_inputTensor, texTransform);
            m_worker.Schedule(m_inputTensor);
            yield return null;

            // Kick off both readbacks in parallel.
            var detAwaiter = (m_worker.PeekOutput(0) as Tensor<float>)
                .ReadbackAndCloneAsync()
                .GetAwaiter();
            var protoAwaiter = (m_worker.PeekOutput(1) as Tensor<float>)
                .ReadbackAndCloneAsync()
                .GetAwaiter();

            while (!detAwaiter.IsCompleted || !protoAwaiter.IsCompleted)
                yield return null;

            using var detTensor = detAwaiter.GetResult();
            using var protoTensor = protoAwaiter.GetResult();

            // detTensor : (1, rows, N)  where rows = 4 + numClasses + 32
            // Sentis may transpose to (1, N, rows) depending on export — check shape.
            // Ultralytics ONNX with nms=False exports as (1, rows, N).
            int rows = detTensor.shape[1]; // 4 + numClasses + MaskCoeffs
            int N = detTensor.shape[2]; // number of anchors (8400 for 640px input)

            if (!m_dimensionsDerived)
            {
                m_dimensionsDerived = true;
                int derivedClasses = rows - 4 - MaskCoeffs;
                if (derivedClasses != m_numClasses)
                {
                    Debug.LogWarning(
                        $"[Seg/Instance] Class count corrected: {m_numClasses} -> {derivedClasses}"
                    );
                    m_numClasses = derivedClasses;
                }
                int protoH = protoTensor.shape[2];
                int protoW = protoTensor.shape[3];
                Debug.Log(
                    $"[Seg/Instance] det=({detTensor.shape[0]},{rows},{N}) "
                        + $"proto=(1,{MaskCoeffs},{protoH},{protoW})"
                );
            }

            int protoRows = protoTensor.shape[2];
            int protoCols = protoTensor.shape[3];

            // ── Score-filter: collect indices for selectedClassId ──────────────
            m_nmsFiltered.Clear();
            for (int i = 0; i < N; i++)
            {
                float score = detTensor[0, 4 + selectedClassId, i];
                if (score >= m_scoreThreshold)
                    m_nmsFiltered.Add(i);
            }

            if (m_nmsFiltered.Count == 0)
            {
                Debug.Log(
                    $"[Seg/Instance] No detections above threshold for class "
                        + $"{selectedClassId} ('{GetLabel(selectedClassId)}')."
                );
                yield break;
            }

            // Sort by score descending.
            m_nmsFiltered.Sort(
                (a, b) =>
                    detTensor[0, 4 + selectedClassId, b]
                        .CompareTo(detTensor[0, 4 + selectedClassId, a])
            );

            // ── NMS ───────────────────────────────────────────────────────────
            if (m_nmsSuppressed.Length < m_nmsFiltered.Count)
                m_nmsSuppressed = new bool[m_nmsFiltered.Count];
            else
                System.Array.Clear(m_nmsSuppressed, 0, m_nmsFiltered.Count);

            // Convert cx,cy,w,h → x1,y1,x2,y2 for IoU.
            System.Func<int, Vector4> getCorners = idx =>
            {
                float cx = detTensor[0, 0, idx];
                float cy = detTensor[0, 1, idx];
                float w = detTensor[0, 2, idx];
                float h = detTensor[0, 3, idx];
                return new Vector4(cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f);
            };

            int bestIdx = -1;
            for (int i = 0; i < m_nmsFiltered.Count; i++)
            {
                if (m_nmsSuppressed[i])
                    continue;

                if (bestIdx < 0)
                    bestIdx = m_nmsFiltered[i]; // highest-score survivor

                Vector4 boxI = getCorners(m_nmsFiltered[i]);
                for (int j = i + 1; j < m_nmsFiltered.Count; j++)
                {
                    if (m_nmsSuppressed[j])
                        continue;
                    if (CalculateIoU(boxI, getCorners(m_nmsFiltered[j])) > m_iouThreshold)
                        m_nmsSuppressed[j] = true;
                }
            }

            if (bestIdx < 0)
            {
                Debug.Log("[Seg/Instance] All detections suppressed by NMS.");
                yield break;
            }

            // ── Decode mask for bestIdx ───────────────────────────────────────
            // coeffs: float[MaskCoeffs]
            // mask_raw[r,c] = sigmoid( sum_k( coeffs[k] * proto[k,r,c] ) )
            // threshold at m_maskThreshold → binary mask

            // Read the 32 coefficients for this detection.
            int coeffOffset = 4 + m_numClasses;
            var coeffs = new float[MaskCoeffs];
            for (int k = 0; k < MaskCoeffs; k++)
                coeffs[k] = detTensor[0, coeffOffset + k, bestIdx];

            // Build the class map at proto resolution.
            // Foreground pixel = selectedClassId, background = 0.
            var classMap = new int[protoRows * protoCols];

            for (int row = 0; row < protoRows; row++)
            for (int col = 0; col < protoCols; col++)
            {
                float dot = 0f;
                for (int k = 0; k < MaskCoeffs; k++)
                    dot += coeffs[k] * protoTensor[0, k, row, col];

                // Sigmoid
                float maskVal = 1f / (1f + Mathf.Exp(-dot));
                classMap[row * protoCols + col] = maskVal >= m_maskThreshold ? selectedClassId : 0;
            }

            // ── Crop mask to the detection bounding box ───────────────────────
            // The detection bbox is in model-input pixel space (0..m_inputW/H).
            // Scale it to proto space.
            Vector4 bboxInput = getCorners(bestIdx);
            float scaleX = (float)protoCols / m_inputW;
            float scaleY = (float)protoRows / m_inputH;

            int bx1 = Mathf.Clamp(Mathf.FloorToInt(bboxInput.x * scaleX), 0, protoCols - 1);
            int by1 = Mathf.Clamp(Mathf.FloorToInt(bboxInput.y * scaleY), 0, protoRows - 1);
            int bx2 = Mathf.Clamp(Mathf.CeilToInt(bboxInput.z * scaleX), 0, protoCols);
            int by2 = Mathf.Clamp(Mathf.CeilToInt(bboxInput.w * scaleY), 0, protoRows);

            // Zero out pixels outside the bounding box.
            for (int row = 0; row < protoRows; row++)
            {
                bool inRow = (row >= by1 && row < by2);
                for (int col = 0; col < protoCols; col++)
                {
                    if (!inRow || col < bx1 || col >= bx2)
                        classMap[row * protoCols + col] = 0;
                }
            }

            // Guard: at least one foreground pixel.
            bool found = false;
            for (int i = 0; i < classMap.Length; i++)
                if (classMap[i] == selectedClassId)
                {
                    found = true;
                    break;
                }

            if (!found)
            {
                Debug.Log(
                    $"[Seg/Instance] Decoded mask for class {selectedClassId} "
                        + "is empty after bbox crop."
                );
                yield break;
            }

            FireResult(classMap, protoCols, protoRows, selectedClassId, cropInfo);
        }

        // ── Shared result fire ────────────────────────────────────────────────
        private void FireResult(
            int[] classMap,
            int mapW,
            int mapH,
            int selectedClassId,
            CropInfo cropInfo
        )
        {
            var result = new SegResult
            {
                classMap = classMap,
                mapW = mapW,
                mapH = mapH,
                targetClassId = selectedClassId,
                label = GetLabel(selectedClassId),
                cropViewport = cropInfo.ViewportNorm,
                modelInputUVRect = cropInfo.ModelInputUVRect,
                isPlaceholder = false,
            };

            Debug.Log(
                $"[Seg] Done - mode={(m_useInstanceSegmentation ? "instance" : "semantic")} "
                    + $"class='{result.label}' ({selectedClassId}) "
                    + $"mapSize={mapW}x{mapH}"
            );
            OnSegmentationReady?.Invoke(result, m_cropExtractor.CachedCameraPose);
        }

        // ── IoU (same formula as MultiDetectionRunManager) ────────────────────
        private static float CalculateIoU(Vector4 a, Vector4 b)
        {
            float x1 = Mathf.Max(a.x, b.x),
                y1 = Mathf.Max(a.y, b.y);
            float x2 = Mathf.Min(a.z, b.z),
                y2 = Mathf.Min(a.w, b.w);
            float inter = Mathf.Max(0, x2 - x1) * Mathf.Max(0, y2 - y1);
            float areaA = (a.z - a.x) * (a.w - a.y);
            float areaB = (b.z - b.x) * (b.w - b.y);
            float union = areaA + areaB - inter;
            return union == 0 ? 0 : inter / union;
        }

        // ── Label helper ──────────────────────────────────────────────────────
        private string GetLabel(int id) =>
            m_labelNames != null && id < m_labelNames.Length
                ? m_labelNames[id].Trim()
                : $"class_{id}";
    }
}
