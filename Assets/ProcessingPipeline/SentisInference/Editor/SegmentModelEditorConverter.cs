using CVMQ3.ProcessingPipeline;
using Meta.XR.Samples;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace CVMQ3.ProcessingPipeline.Editor
{
    [MetaCodeSample("ProcessingPipeline")]
    [CustomEditor(typeof(SegmentationRunManager))]
    public class SegmentModelEditorConverter : UnityEditor.Editor
    {
        private const string FILEPATH_SEM =
            "Assets/ProcessingPipeline/SentisInference/Models/sem_sentis.sentis";

        private const string FILEPATH_SEG =
            "Assets/ProcessingPipeline/SentisInference/Models/seg_sentis.sentis";

        private SegmentationRunManager m_targetClass;

        public void OnEnable()
        {
            m_targetClass = (SegmentationRunManager)target;
        }

        public override void OnInspectorGUI()
        {
            _ = DrawDefaultInspector();

            EditorGUILayout.Space(8);

            if (GUILayout.Button("Convert Semantic model (YOLO26n-sem) to Sentis"))
            {
                OnEnable();
                ConvertModel(m_targetClass.OnnxModel, FILEPATH_SEM, "semantic");
            }

            EditorGUILayout.Space(4);

            if (GUILayout.Button("Convert Instance Segmentation model (YOLO26n-seg) to Sentis"))
            {
                OnEnable();
                ConvertModel(m_targetClass.OnnxModel, FILEPATH_SEG, "instance");
            }
        }

        private static void ConvertModel(ModelAsset onnxModel, string outputPath, string label)
        {
            if (onnxModel == null)
            {
                Debug.LogError(
                    $"[SegConverter] OnnxModel is not assigned — "
                        + $"cannot convert {label} model."
                );
                return;
            }

            var model = ModelLoader.Load(onnxModel);
            ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model);
            ModelWriter.Save(outputPath, model);
            AssetDatabase.Refresh();

            Debug.Log($"[SegConverter] Saved {label} Sentis model → {outputPath}");
        }
    }
}