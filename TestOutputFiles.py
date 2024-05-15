import unittest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
from GNNProcessor import GNSSProcessor

class TestOutputFiles(unittest.TestCase):

    @patch('GNNProcessor.os.path.exists')
    def test_file_existence(self, mock_exists):
        mock_exists.return_value = True
        processor = GNSSProcessor(input_filepath='input_logs/fake_file.txt')
        self.assertEqual(processor.input_filepath, 'input_logs/fake_file.txt')

    def test_read_input_data(self):
        processor = GNSSProcessor(input_filepath='input_logs/driving.txt')
        processor.read_input_data()
        self.assertIsInstance(processor.android_fixes, pd.DataFrame)
        self.assertIsInstance(processor.measurements, pd.DataFrame)

    def test_preprocess_data(self):
        processor = GNSSProcessor(input_filepath='input_logs/driving.txt')
        processor.read_input_data()
        processor.preprocess_data()
        self.assertIn('satPRN', processor.measurements.columns)
        self.assertIn('UnixTime', processor.measurements.columns)

    def test_create_kml_file(self):
        coords = [(37.7749, -122.4194, 10), (34.0522, -118.2437, 20)]
        output_file = 'test_coordinates.kml'
        GNSSProcessor.create_kml_file(coords, output_file)
        self.assertTrue(os.path.exists(output_file))
        os.remove(output_file)

if __name__ == '__main__':
    unittest.main()