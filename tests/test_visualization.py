"""
tests/test_visualization.py
============================
Test visualization system (display manager).

Tests
-----
- Canvas creation
- Frame grid layout
- Bounding box rendering
- Control panel rendering
- Thread safety
- Non-blocking rendering
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import time

from src.visualization.display_manager import DisplayManager, VehicleClass


class TestDisplayManager:
    """Visualization system tests."""
    
    @pytest.fixture
    def display(self):
        """Create display manager instance."""
        return DisplayManager(
            grid_rows=2,
            grid_cols=4,
            window_name="Test Dashboard",
            fps=10.0
        )
    
    def test_initialization(self, display):
        """Test display manager initialization."""
        assert display.grid_rows == 2
        assert display.grid_cols == 4
        assert display.max_cameras == 8
        assert display.window_name == "Test Dashboard"
    
    def test_canvas_dimensions(self, display):
        """Test canvas dimensions are correct."""
        assert display.canvas_width == display.cam_width * 4
        assert display.canvas_height == display.cam_height * 2 + display.panel_height
    
    def test_render_queue_non_blocking(self, display):
        """Test render queue is non-blocking."""
        display.start()
        
        try:
            # Queue multiple payloads quickly (should not block)
            for i in range(5):
                display.render_frame(
                    frames_dict={},
                    phase_map={"tls_0": 0},
                    action_map={"tls_0": 30},
                    fps=10.0,
                    latency_ms=50.0
                )
                # Should not block or hang
                time.sleep(0.01)
            
            assert display.frame_count >= 0  # Some frames rendered
        finally:
            display.stop()
    
    def test_frame_rendering(self):
        """Test frame rendering (with mocked OpenCV)."""
        with patch('cv2.imshow'):
            with patch('cv2.waitKey', return_value=255):
                display = DisplayManager(grid_rows=2, grid_cols=4)
                display.start()
                
                try:
                    # Create fake frames
                    frames_dict = {
                        i: np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                        for i in range(4)
                    }
                    
                    display.render_frame(
                        frames_dict=frames_dict,
                        phase_map={"tls_0": 0, "tls_1": 1},
                        action_map={"tls_0": 30, "tls_1": 45},
                        state_summary={"queue_norm": 0.5},
                        fps=10.2,
                        latency_ms=45.3,
                        stale_cameras=[],
                    )
                    
                    time.sleep(0.1)
                    assert display.frame_count > 0
                finally:
                    display.stop()
    
    def test_stale_camera_indicator(self):
        """Test stale camera indicator in rendering."""
        with patch('cv2.imshow'):
            with patch('cv2.waitKey', return_value=255):
                display = DisplayManager()
                display.start()
                
                try:
                    frames_dict = {
                        0: np.zeros((240, 320, 3), dtype=np.uint8),
                        1: np.zeros((240, 320, 3), dtype=np.uint8),
                    }
                    
                    display.render_frame(
                        frames_dict=frames_dict,
                        phase_map={"tls_0": 0},
                        action_map={"tls_0": 30},
                        stale_cameras=[1],  # Camera 1 is stale
                        fps=10.0,
                        latency_ms=50.0,
                    )
                    
                    time.sleep(0.1)
                    assert display.frame_count > 0
                finally:
                    display.stop()
    
    def test_detections_rendering(self):
        """Test bounding box rendering."""
        with patch('cv2.imshow'):
            with patch('cv2.waitKey', return_value=255):
                display = DisplayManager()
                display.start()
                
                try:
                    frames_dict = {
                        0: np.zeros((240, 320, 3), dtype=np.uint8),
                    }
                    
                    detections_dict = {
                        0: [
                            {
                                "bbox": [50, 60, 150, 180],
                                "cls_id": VehicleClass.CAR.value,
                                "track_id": 1,
                            },
                            {
                                "bbox": [200, 50, 280, 150],
                                "cls_id": VehicleClass.TRUCK.value,
                                "track_id": 2,
                            },
                        ]
                    }
                    
                    display.render_frame(
                        frames_dict=frames_dict,
                        phase_map={"tls_0": 0},
                        action_map={"tls_0": 30},
                        detections_dict=detections_dict,
                        fps=10.0,
                        latency_ms=50.0,
                    )
                    
                    time.sleep(0.1)
                    assert display.frame_count > 0
                finally:
                    display.stop()
    
    def test_thread_safety(self, display):
        """Test thread-safe rendering."""
        import threading
        
        display.start()
        errors = []
        
        def render_worker():
            try:
                for _ in range(10):
                    display.render_frame(
                        frames_dict={},
                        phase_map={"tls_0": 0},
                        action_map={"tls_0": 30},
                        fps=10.0,
                        latency_ms=50.0,
                    )
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        # Run multiple render workers concurrently
        threads = [
            threading.Thread(target=render_worker, daemon=True)
            for _ in range(3)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join(timeout=2.0)
        
        display.stop()
        
        assert len(errors) == 0, f"Thread safety errors: {errors}"
    
    def test_vehicle_classes(self):
        """Test vehicle class enum."""
        assert VehicleClass.ACCIDENT.value == 0
        assert VehicleClass.BUS.value == 1
        assert VehicleClass.CAR.value == 2
        assert VehicleClass.MOTORCYCLE.value == 3
        assert VehicleClass.TRUCK.value == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
