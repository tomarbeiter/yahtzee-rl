from setuptools import setup

setup(name='yahtzee_rl',
      version='0.1.0',
      description='Different agents/envs/algorithms developed to learn how to play Yahtzee.',
      long_description='',
      url='https://github.com/tomarbeiter/yahtzee_rl',
      author='Tom Arbeiter',
      license='Apache 2.0',
      packages=['yahtzee_agents'],
      install_requires=['pytest', 'sphinx'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        'Intended Audience :: Developers',
      ],
      )
      