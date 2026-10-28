import json
import unittest
import numpy as np
from vision_worker import decode_detections


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.prompt = {'name':'纸杯','texts':['paper cup','bottle'],'positive_count':1}

    def test_negative_class_and_overlapping_boxes_are_filtered(self):
        output = np.array([[[40,41,150],[60,61,60],[20,20,20],[40,40,40],
                            [.8,.7,.6],[.1,.1,.9]]],dtype=np.float32)
        result = decode_detections(output,self.prompt,.5,(0,10),(100,200,3))
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['box'],[60.,60.,100.,100.])
        self.assertEqual(result[0]['score'],.8)
        json.dumps(result,ensure_ascii=False,allow_nan=False)

    def test_probabilities_are_not_sigmoid_transformed_twice(self):
        output = np.array([[[40],[60],[20],[40],[.05],[.01]]],dtype=np.float32)
        self.assertEqual(decode_detections(output,self.prompt,1.,(0,0),(100,200,3)),[])
        output[0,4,0]=.21
        self.assertEqual(decode_detections(output,self.prompt,1.,(0,0),(100,200,3))[0]['score'],.21)

    def test_invalid_output_is_an_error(self):
        output = np.zeros((1,6,1),dtype=np.float32);output[0,4,0]=float('nan')
        with self.assertRaises(RuntimeError):
            decode_detections(output,self.prompt,1.,(0,0),(100,200,3))


if __name__ == '__main__':unittest.main()
