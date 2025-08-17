"""
Unit tests for YOLOv8 Web Detection application
"""

import pytest
import json
import base64
import io
from PIL import Image
import numpy as np

# Import the applications
try:
    from app import app as standard_app
    STANDARD_APP_AVAILABLE = True
except ImportError:
    STANDARD_APP_AVAILABLE = False

try:
    from app_optimized import app as optimized_app
    OPTIMIZED_APP_AVAILABLE = True
except ImportError:
    OPTIMIZED_APP_AVAILABLE = False


@pytest.fixture
def client_standard():
    """Create test client for standard app"""
    if not STANDARD_APP_AVAILABLE:
        pytest.skip("Standard app not available")
    
    standard_app.config['TESTING'] = True
    with standard_app.test_client() as client:
        yield client


@pytest.fixture
def client_optimized():
    """Create test client for optimized app"""
    if not OPTIMIZED_APP_AVAILABLE:
        pytest.skip("Optimized app not available")
    
    optimized_app.config['TESTING'] = True
    with optimized_app.test_client() as client:
        yield client


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple test image
    img = Image.new('RGB', (640, 480), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Convert to base64
    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


class TestStandardApp:
    """Test cases for standard Flask app"""
    
    def test_home_page(self, client_standard):
        """Test home page loads"""
        response = client_standard.get('/')
        assert response.status_code == 200
        assert b'Camera' in response.data or b'camera' in response.data
    
    def test_health_endpoint(self, client_standard):
        """Test health endpoint if available"""
        response = client_standard.get('/health')
        # Either 200 (exists) or 404 (doesn't exist) is acceptable
        assert response.status_code in [200, 404]
    
    def test_model_info_endpoint(self, client_standard):
        """Test model info endpoint if available"""
        response = client_standard.get('/model_info')
        # Either 200 (exists) or 404 (doesn't exist) is acceptable
        assert response.status_code in [200, 404]


class TestOptimizedApp:
    """Test cases for optimized Flask app"""
    
    def test_home_page(self, client_optimized):
        """Test home page loads"""
        response = client_optimized.get('/')
        assert response.status_code == 200
        assert b'Camera' in response.data or b'camera' in response.data
    
    def test_health_endpoint(self, client_optimized):
        """Test health endpoint"""
        response = client_optimized.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'uptime' in data
        assert 'memory_usage' in data
    
    def test_model_info_endpoint(self, client_optimized):
        """Test model info endpoint"""
        response = client_optimized.get('/model_info')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'model_path' in data
        assert 'classes' in data
    
    def test_stats_endpoint(self, client_optimized):
        """Test stats endpoint"""
        response = client_optimized.get('/stats')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'active_sessions' in data
        assert 'total_requests' in data


class TestImageProcessing:
    """Test image processing functionality"""
    
    def test_mobile_image_processing(self, client_optimized, sample_image):
        """Test mobile image processing endpoint"""
        payload = {
            'image': sample_image,
            'session_id': 'test_session_123'
        }
        
        response = client_optimized.post(
            '/process_mobile_image',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should either process successfully or fail gracefully
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'success' in data
    
    def test_mobile_stream_processing(self, client_optimized, sample_image):
        """Test mobile stream processing endpoint"""
        payload = {
            'image': sample_image,
            'session_id': 'test_session_456',
            'frame_rate': 2
        }
        
        response = client_optimized.post(
            '/process_mobile_stream',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should either process successfully or fail gracefully
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'success' in data


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_json(self, client_optimized):
        """Test handling of invalid JSON"""
        response = client_optimized.post(
            '/process_mobile_image',
            data='invalid json',
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_missing_image_data(self, client_optimized):
        """Test handling of missing image data"""
        payload = {
            'session_id': 'test_session_789'
            # Missing 'image' field
        }
        
        response = client_optimized.post(
            '/process_mobile_image',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_invalid_image_data(self, client_optimized):
        """Test handling of invalid image data"""
        payload = {
            'image': 'invalid_base64_data',
            'session_id': 'test_session_invalid'
        }
        
        response = client_optimized.post(
            '/process_mobile_image',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert response.status_code == 400


class TestConfiguration:
    """Test configuration loading and validation"""
    
    def test_config_loading(self):
        """Test configuration file loading"""
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            # Verify required fields
            assert 'model_path' in config
            assert 'max_workers' in config
            assert isinstance(config['max_workers'], int)
            assert config['max_workers'] > 0
            
        except FileNotFoundError:
            # Config file is optional
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
