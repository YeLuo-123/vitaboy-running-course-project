from glob import glob
from setuptools import find_packages, setup

setup(name='vitaboy_navigation', version='0.1.0', packages=find_packages(),
      data_files=[('share/ament_index/resource_index/packages', ['resource/vitaboy_navigation']),
                  ('share/vitaboy_navigation', ['package.xml', 'LICENSE', 'THIRD_PARTY_HONOR_LICENSE', 'provenance.json']),
                  ('share/vitaboy_navigation/config', glob('config/*')),
                  ('share/vitaboy_navigation/launch', glob('launch/*.py'))],
      install_requires=['setuptools'], license='BSD-3-Clause',
      description='VitaBoy simulation adapters for the G1-Nav SLAM/navigation front-end',
      maintainer='Course project', maintainer_email='course@example.invalid',
      entry_points={'console_scripts': [
          'bridge = vitaboy_navigation.bridge:main',
          'planner = vitaboy_navigation.navigation:planner_main',
          'tracker = vitaboy_navigation.navigation:tracker_main',
          'check_graph = vitaboy_navigation.check_graph:main']})
