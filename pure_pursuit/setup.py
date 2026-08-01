from setuptools import find_packages, setup

package_name = 'pure_pursuit'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
   entry_points={
    'console_scripts': [
        'waypoint_recorder = pure_pursuit.waypoint_recorder:main',
        'pure_pursuit_node  = pure_pursuit.pure_pursuit_node:main',
        'lqr_node           = pure_pursuit.lqr_node:main',
        'mpc_node          = pure_pursuit.mpc_node:main',
        'hybrid_controller = pure_pursuit.hybrid_controller:main',
        'stanley_controller = pure_pursuit.stanley_controller:main',
    ],
},
)
