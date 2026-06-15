using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Meta.XR;
using Meta.XR.Samples;
using Unity.Collections;
using Unity.InferenceEngine;
using UnityEngine;

namespace CVMQ3.ProcessingPipeline
{
    [MetaCodeSample("MultiObjectDetection")]
    public class MultiDetectionRunManager : MonoBehaviour
    {
        [SerializeField]
        private PassthroughCameraAccess m_cameraAccess;

        [Header("Sentis Model config")]
        [SerializeField]
        private BackendType m_backend = BackendType.GPUCompute;

        [SerializeField]
        private ModelAsset m_sentisModel;

        [SerializeField]
        private TextAsset m_labelsAsset;

        [SerializeField, Range(0, 1)]
        private float m_iouThreshold = 0.6f;

        [SerializeField, Range(0, 1)]
        private float m_scoreThreshold = 0.45f;

        [Header("UI display references")]
        [SerializeField]
        private MultiDetectionUiManager m_uiInference;

        [Header("[Editor Only] Convert to Sentis")]
        public ModelAsset OnnxModel;

        [Space(40)]
        private Worker m_engine;
        private Vector2Int m_inputSize;

        private Tensor<float> m_inputTensor;

        private TextureTransform m_textureTransform;

        private readonly List<(int classId, Vector4 boundingBox)> m_detections =
            new List<(int classId, Vector4 boundingBox)>();

        private readonly List<int> m_nmsFilteredIndices = new List<int>();
        private bool[] m_nmsSuppressed = new bool[0];

        public System.Action OnInferenceStart;
        public System.Action OnInferenceComplete;

        private void Awake()
        {
            var model = ModelLoader.Load(m_sentisModel);
            var inputShape = model.inputs[0].shape;
            m_inputSize = new Vector2Int(inputShape.Get(2), inputShape.Get(3));
            m_engine = new Worker(model, m_backend);

            m_inputTensor = new Tensor<float>(new TensorShape(1, 3, m_inputSize.x, m_inputSize.y));
            m_textureTransform = new TextureTransform().SetDimensions(0, 0, 3);
        }

        private IEnumerator Start()
        {
            m_uiInference.SetLabels(m_labelsAsset);
            while (true)
            {
                yield return RunInference();
            }
        }

        private void OnDestroy()
        {
            m_inputTensor?.Dispose();
            m_engine?.PeekOutput(0)?.CompleteAllPendingOperations();
            m_engine?.PeekOutput(1)?.CompleteAllPendingOperations();
            m_engine?.PeekOutput(2)?.CompleteAllPendingOperations();
            m_engine?.Dispose();
        }

        internal static void PreloadModel(ModelAsset modelAsset)
        {
            var model = ModelLoader.Load(modelAsset);
            var inputShape = model.inputs[0].shape;

            using var worker = new Worker(model, BackendType.GPUCompute); // match runtime backend
            Texture tempTexture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            var textureTransform = new TextureTransform().SetDimensions(
                tempTexture.width,
                tempTexture.height,
                3
            );
            using var input = new Tensor<float>(
                new TensorShape(1, 3, inputShape.Get(2), inputShape.Get(3))
            );
            TextureConverter.ToTensor(tempTexture, input, textureTransform);
            worker.Schedule(input);
            worker.PeekOutput(0).CompleteAllPendingOperations();
            worker.PeekOutput(1).CompleteAllPendingOperations();
            worker.PeekOutput(2).CompleteAllPendingOperations();
            Destroy(tempTexture);
        }

        private IEnumerator RunInference()
        {
            OnInferenceStart?.Invoke();

            if (!m_cameraAccess.IsPlaying)
            {
                yield break;
            }

            [DllImport("OVRPlugin", CallingConvention = CallingConvention.Cdecl)]
            static extern OVRPlugin.Result ovrp_GetNodePoseStateAtTime(
                double time,
                OVRPlugin.Node nodeId,
                out OVRPlugin.PoseStatef nodePoseState
            );

            if (
                !ovrp_GetNodePoseStateAtTime(
                        OVRPlugin.GetTimeInSeconds(),
                        OVRPlugin.Node.Head,
                        out _
                    )
                    .IsSuccess()
            )
            {
                Debug.LogWarning("[Detect] ovrp_GetNodePoseStateAtTime failed — skipping frame");
                yield break;
            }

            var cachedCameraPose = m_cameraAccess.GetCameraPose();
            Texture targetTexture = m_cameraAccess.GetTexture();
            Debug.Log(
                $"[Detect] Camera texture size: {targetTexture.width}x{targetTexture.height}"
            );
            TextureConverter.ToTensor(targetTexture, m_inputTensor, m_textureTransform);

            m_engine.Schedule(m_inputTensor);

            var boxesAwaiter = (m_engine.PeekOutput(0) as Tensor<float>)
                .ReadbackAndCloneAsync()
                .GetAwaiter();
            var classIDsAwaiter = (m_engine.PeekOutput(1) as Tensor<int>)
                .ReadbackAndCloneAsync()
                .GetAwaiter();
            var scoresAwaiter = (m_engine.PeekOutput(2) as Tensor<float>)
                .ReadbackAndCloneAsync()
                .GetAwaiter();

            // Yield until ALL three are ready in parallel.
            while (
                !boxesAwaiter.IsCompleted
                || !classIDsAwaiter.IsCompleted
                || !scoresAwaiter.IsCompleted
            )
            {
                yield return null;
            }

            using var boxes = boxesAwaiter.GetResult();
            using var classIDs = classIDsAwaiter.GetResult();
            using var scores = scoresAwaiter.GetResult();

            if (boxes.shape[0] == 0 || classIDs.shape[0] == 0 || scores.shape[0] == 0)
            {
                Debug.Log("[Detect] Empty output tensors — no detections this frame");
                yield break;
            }

            NonMaxSuppression(
                m_detections,
                boxes,
                classIDs,
                scores,
                m_iouThreshold,
                m_scoreThreshold,
                m_nmsFilteredIndices,
                ref m_nmsSuppressed
            );

            OnInferenceComplete?.Invoke();
            m_uiInference.DrawUIBoxes(m_detections, m_inputSize, cachedCameraPose);
        }

        private static void NonMaxSuppression(
            List<(int classId, Vector4 boundingBox)> outDetections,
            Tensor<float> boxes,
            Tensor<int> classIDs,
            Tensor<float> scores,
            float iouThreshold,
            float scoreThreshold,
            List<int> filteredIndices, // pre-allocated, will be Cleared
            ref bool[] suppressed // pre-allocated, grown as needed
        )
        {
            outDetections.Clear();
            filteredIndices.Clear();

            NativeArray<float>.ReadOnly scoresArray = scores.AsReadOnlyNativeArray();

            for (int i = 0; i < scoresArray.Length; i++)
            {
                if (scoresArray[i] >= scoreThreshold)
                    filteredIndices.Add(i);
            }

            if (filteredIndices.Count == 0)
                return;

            // Grow the suppressed array if needed — avoids allocation in the common case.
            if (suppressed.Length < filteredIndices.Count)
                suppressed = new bool[filteredIndices.Count];
            else
                System.Array.Clear(suppressed, 0, filteredIndices.Count);

            filteredIndices.Sort((a, b) => scoresArray[b].CompareTo(scoresArray[a]));

            for (int i = 0; i < filteredIndices.Count; i++)
            {
                if (suppressed[i])
                    continue;

                int idx = filteredIndices[i];
                outDetections.Add((classIDs[idx], GetBox(boxes, idx)));

                for (int j = i + 1; j < filteredIndices.Count; j++)
                {
                    if (suppressed[j])
                        continue;
                    int jdx = filteredIndices[j];
                    if (CalculateIoU(GetBox(boxes, idx), GetBox(boxes, jdx)) > iouThreshold)
                        suppressed[j] = true;
                }
            }
        }

        private static Vector4 GetBox(Tensor<float> boxes, int i) =>
            new Vector4(boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3]);

        internal static float CalculateIoU(Vector4 boxA, Vector4 boxB)
        {
            float x1 = Mathf.Max(boxA.x, boxB.x);
            float y1 = Mathf.Max(boxA.y, boxB.y);
            float x2 = Mathf.Min(boxA.z, boxB.z);
            float y2 = Mathf.Min(boxA.w, boxB.w);

            float intersectionWidth = Mathf.Max(0, x2 - x1);
            float intersectionHeight = Mathf.Max(0, y2 - y1);
            float intersectionArea = intersectionWidth * intersectionHeight;

            float boxAArea = (boxA.z - boxA.x) * (boxA.w - boxA.y);
            float boxBArea = (boxB.z - boxB.x) * (boxB.w - boxB.y);
            float unionArea = boxAArea + boxBArea - intersectionArea;

            return unionArea == 0 ? 0 : intersectionArea / unionArea;
        }
    }
}
