// Copyright (c) Meta Platforms, Inc. and affiliates.
using CVMQ3.ProcessingPipeline;
using Meta.XR.Samples;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace CVMQ3.ProcessingPipeline.Editor
{
    [MetaCodeSample("ProcessingPipeline")]
    [CustomEditor(typeof(MultiDetectionRunManager))]
    public class DetectionModelEditorConverter : UnityEditor.Editor
    {
        private const string FILEPATH =
            "Assets/ProcessingPipeline/SentisInference/Models/detect_sentis.sentis";
        private MultiDetectionRunManager m_targetClass;
        private float m_iouThreshold;
        private float m_scoreThreshold;

        public void OnEnable()
        {
            m_targetClass = (MultiDetectionRunManager)target;
            m_iouThreshold = serializedObject.FindProperty("m_iouThreshold").floatValue;
            m_scoreThreshold = serializedObject.FindProperty("m_scoreThreshold").floatValue;
        }

        public override void OnInspectorGUI()
        {
            _ = DrawDefaultInspector();

            if (GUILayout.Button("Generate Yolov8n Sentis model with Non-Max-Supression layer"))
            {
                OnEnable(); // Get the latest values from the serialized object
                ConvertModel(); // convert the ONNX model to sentis
            }
        }

        private void ConvertModel()
        {
            //Load model
            var model = ModelLoader.Load(m_targetClass.OnnxModel);

            //Here we transform the output of the model by feeding it through a Non-Max-Suppression layer.
            var graph = new FunctionalGraph();
            var input = graph.AddInput(model, 0);

            var centersToCornersData = new[]
            {
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                -0.5f,
                0,
                0.5f,
                0,
                0,
                -0.5f,
                0,
                0.5f,
            };
            var centersToCorners = Functional.Constant(new TensorShape(4, 4), centersToCornersData);
            var modelOutput = Functional.Forward(model, input)[0]; //shape(1,N,85)
            // Following for yolo model. in (1, 84, N) out put shape
            var boxCoords = modelOutput[0, ..4, ..].Transpose(0, 1);
            var allScores = modelOutput[0, 4.., ..].Transpose(0, 1);
            var scores = Functional.ReduceMax(allScores, 1); //shape=(N)
            var classIDs = Functional.ArgMax(allScores, 1); //shape=(N)
            var corners = Functional.MatMul(boxCoords, centersToCorners); //shape=(N,4)
            var modelFinal = graph.Compile(corners, classIDs, scores);

            //Export the model to Sentis format
            ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref modelFinal);
            ModelWriter.Save(FILEPATH, modelFinal);

            // refresh assets
            AssetDatabase.Refresh();
        }
    }
}
