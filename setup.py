from setuptools import setup
import os
from glob import glob

package_name = 'fire_fly'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gary',
    maintainer_email='lvganglvgang@gmail.com',
    description='ROS2 WebRTC video streaming package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'video_sender = fire_fly.video_sender:main',
        ],
    },
)
