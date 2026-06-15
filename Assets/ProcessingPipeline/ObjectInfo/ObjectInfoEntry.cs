// ObjectInfoEntry.cs
using System;
using System.Collections.Generic;

namespace CVMQ3.ProcessingPipeline
{
    [Serializable]
    public class ObjectInfoEntry
    {
        public int id;
        public string name;
        public List<string> aliases;
        public string category;
        public string supercategory;
        public string occlusion_risk;
        public string pose_estimation_notes;
        public List<string> material;
        public SegmentationInfo segmentation;
        public ShapeProfile shape_profile;
        public KeypointsInfo keypoints;
        public List<int> color_hint_rgb;
        public TypicalDimensions typical_dimensions_cm;

        [Serializable]
        public class SegmentationInfo
        {
            public string difficulty;
            public string reason;
            public string recommended_approach;
            public string mask_type;
        }

        [Serializable]
        public class ShapeProfile
        {
            public string geometry;
            public string symmetry;
            public float? typical_aspect_ratio_h_w;
            public bool? hollow;
        }

        [Serializable]
        public class KeypointsInfo
        {
            public List<string> suggested;
            public int count;
        }

        [Serializable]
        public class TypicalDimensions
        {
            public List<float> height;
            public List<float> diameter;
            public List<float> width;
            public List<float> depth;
        }
    }
}