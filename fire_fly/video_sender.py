#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import asyncio
import json
import logging
import websockets
from threading import Thread
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class VideoSender(Node):
    def __init__(self):
        super().__init__('video_sender')
        self.bridge = CvBridge()
        
        # Parameters
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('websocket_url', 'ws://localhost:8000/ws/video')
        self.declare_parameter('fps', 30)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('jpeg_quality', 85)
        
        # Get parameters
        self.camera_topic = self.get_parameter('camera_topic').value
        self.websocket_url = self.get_parameter('websocket_url').value
        self.fps = self.get_parameter('fps').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        
        # Initialize CV bridge and websocket client
        self.bridge = CvBridge()
        self.websocket = None
        self.is_connected = False
        self.loop = None
        self.frame_queue = asyncio.Queue()
        
        # Create subscription to camera topic
        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10)
        
        logger.info(f"Subscribing to camera topic: {self.camera_topic}")
        logger.info(f"WebSocket URL: {self.websocket_url}")
        
        # Start websocket connection in a separate thread
        self.ws_thread = Thread(target=self.start_websocket_client)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        logger.info(
            f'Video sender initialized.\n'
            f'Listening on topic: {self.camera_topic}\n'
            f'Image size: {self.image_width}x{self.image_height}\n'
            f'Target FPS: {self.fps}'
        )

    def image_callback(self, msg):
        """Callback for processing incoming image messages."""
        if not self.is_connected:
            logger.warning("Not connected to WebSocket server, skipping frame")
            return
            
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Resize image if needed
            if cv_image.shape[1] != self.image_width or cv_image.shape[0] != self.image_height:
                cv_image = cv2.resize(cv_image, (self.image_width, self.image_height))
            
            # Convert to JPEG for efficient transmission
            _, jpeg_frame = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            
            # Add frame to queue
            if self.loop and self.is_connected:
                asyncio.run_coroutine_threadsafe(
                    self.frame_queue.put(jpeg_frame.tobytes()),
                    self.loop
                )
                
        except Exception as e:
            logger.error(f'Error processing image: {str(e)}')

    async def send_frames(self, websocket):
        """Send frames from queue to websocket."""
        while True:
            try:
                frame_data = await self.frame_queue.get()
                if isinstance(frame_data, bytes):
                    await websocket.send(frame_data)
                else:
                    logger.warning(f"Received non-bytes frame data: {type(frame_data)}")
            except Exception as e:
                logger.error(f'Error sending frame: {str(e)}')
                self.is_connected = False
                break

    async def websocket_client(self):
        """Maintain websocket connection."""
        while rclpy.ok():
            try:
                logger.info(f"Attempting to connect to WebSocket server at {self.websocket_url}")
                try:
                    async with websockets.connect(
                        self.websocket_url,
                        ping_interval=None,  # Disable automatic ping since we handle it manually
                        ping_timeout=None,
                        close_timeout=2.0,
                        max_size=None  # Allow large messages
                    ) as websocket:
                        self.websocket = websocket
                        self.is_connected = True
                        logger.info('Connected to WebSocket server')
                        
                        # Create tasks for sending frames and handling messages
                        send_frames_task = asyncio.create_task(self.send_frames(websocket))
                        handle_messages_task = asyncio.create_task(self.handle_messages(websocket))
                        
                        # Wait for either task to complete
                        done, pending = await asyncio.wait(
                            [send_frames_task, handle_messages_task],
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        
                        # Cancel pending tasks
                        for task in pending:
                            task.cancel()
                            
                except websockets.exceptions.InvalidStatusCode as e:
                    logger.error(f"WebSocket connection failed with status code {e.status_code}")
                except websockets.exceptions.InvalidMessage as e:
                    logger.error(f"WebSocket invalid message: {str(e)}")
                except websockets.exceptions.ConnectionClosed as e:
                    logger.error(f"WebSocket connection closed: {str(e)}")
                except Exception as e:
                    logger.error(f'WebSocket connection error: {str(e)}')
                    logger.error(f'Error type: {type(e).__name__}')
                    import traceback
                    logger.error(f'Traceback: {traceback.format_exc()}')
                            
            except Exception as e:
                logger.error(f'Outer WebSocket error: {str(e)}')
                self.is_connected = False
                await asyncio.sleep(5.0)  # Wait before reconnecting

    async def handle_messages(self, websocket):
        """Handle incoming WebSocket messages."""
        try:
            while True:
                message = await websocket.recv()
                if isinstance(message, str):
                    if message == "ping":
                        await websocket.send("pong")
                        logger.debug("Received ping, sent pong")
                    else:
                        logger.debug(f"Received text message: {message}")
                elif isinstance(message, bytes):
                    # This is likely a video frame being echoed back
                    # We can ignore these as they're just our own frames
                    pass
                else:
                    logger.warning(f"Received unexpected message type: {type(message)}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed in message handler")
        except Exception as e:
            logger.error(f"Error handling messages: {str(e)}")

    def start_websocket_client(self):
        """Start websocket client in asyncio event loop."""
        try:
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Run the websocket client
            self.loop.run_until_complete(self.websocket_client())
        except Exception as e:
            logger.error(f"Error in websocket client thread: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

def main(args=None):
    rclpy.init(args=args)
    video_sender = VideoSender()
    
    try:
        rclpy.spin(video_sender)
    except KeyboardInterrupt:
        pass
    finally:
        video_sender.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
