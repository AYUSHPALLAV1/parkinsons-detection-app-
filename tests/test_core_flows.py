import io
import os
import unittest
import uuid

import cv2
import numpy as np
import soundfile as sf

from backend import app as backend_app
from backend.database import DatabaseManager


def build_test_audio_bytes(duration_seconds=1.2, sample_rate=22050):
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 220 * t)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer


def build_test_image_bytes():
    image = np.full((224, 224, 3), 255, dtype=np.uint8)
    cv2.circle(image, (112, 112), 70, (0, 0, 0), 3)
    cv2.line(image, (112, 42), (112, 182), (0, 0, 0), 2)
    success, encoded = cv2.imencode('.png', image)
    if not success:
        raise RuntimeError('Failed to encode test handwriting image.')
    return io.BytesIO(encoded.tobytes())


def build_eye_session(samples=40):
    session = []
    for index in range(samples):
        session.append(
            {
                'timestamp': index * 0.1,
                'gaze_x': 0.5 + 0.02 * np.sin(index / 4.0),
                'gaze_y': 0.5 + 0.01 * np.cos(index / 5.0),
                'target_x': 0.5,
                'target_y': 0.5,
                'ear': 0.3,
                'task': 'Fixation Baseline',
            }
        )
    return session


class CoreFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.join(os.getcwd(), 'tests', '.tmp', str(uuid.uuid4()))
        os.makedirs(cls.temp_dir, exist_ok=True)
        cls.original_db = backend_app.db
        cls.original_testing = backend_app.app.config.get('TESTING', False)

        test_db_path = os.path.join(cls.temp_dir, 'test_parkinsons_detection.db')
        backend_app.db = DatabaseManager(test_db_path)
        backend_app.app.config['TESTING'] = True
        cls.client = backend_app.app.test_client()

    @classmethod
    def tearDownClass(cls):
        backend_app.db = cls.original_db
        backend_app.app.config['TESTING'] = cls.original_testing
        if os.path.exists(cls.temp_dir):
            for file_name in os.listdir(cls.temp_dir):
                os.remove(os.path.join(cls.temp_dir, file_name))
            os.rmdir(cls.temp_dir)

    def test_home_and_dashboard_pages_render(self):
        home_response = self.client.get('/')
        dashboard_response = self.client.get('/dashboard')

        self.assertEqual(home_response.status_code, 200)
        self.assertEqual(dashboard_response.status_code, 200)

    def test_voice_assessment_returns_probability(self):
        response = self.client.post(
            '/api/voice_assessment',
            data={'audio': (build_test_audio_bytes(), 'voice.wav')},
            content_type='multipart/form-data',
        )

        self.assertEqual(response.status_code, 200, response.get_json())
        probability = response.get_json()['probability']
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_handwriting_assessment_returns_probability(self):
        response = self.client.post(
            '/api/handwriting_assessment',
            data={'image': (build_test_image_bytes(), 'spiral.png')},
            content_type='multipart/form-data',
        )

        self.assertEqual(response.status_code, 200, response.get_json())
        probability = response.get_json()['probability']
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_eye_assessment_returns_probability_and_features(self):
        response = self.client.post('/api/eye_assessment', json={'session_data': build_eye_session()})

        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertIn('features', payload)
        self.assertGreaterEqual(payload['probability'], 0.0)
        self.assertLessEqual(payload['probability'], 0.95)

    def test_fusion_ignores_missing_modalities(self):
        response = self.client.post('/api/fuse_results', json={'voice': 0.8, 'eye': None, 'handwriting': 0.4})

        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertAlmostEqual(payload['final_probability'], 0.625, places=6)
        self.assertEqual(payload['verdict'], "Parkinson's Disease Likely")

    def test_fusion_requires_at_least_one_valid_probability(self):
        response = self.client.post('/api/fuse_results', json={'voice': None, 'eye': None, 'handwriting': None})

        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())


if __name__ == '__main__':
    unittest.main()
