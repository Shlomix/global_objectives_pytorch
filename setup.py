from setuptools import setup

setup(name='global_objectives_pytorch',
      version='0.1',
      description='Global Objectives For Pytorch',
      url='https://github.com/Shlomix/global_objectives_pytorch.git',
      author='Shlomi Azoulay',
      author_email='shlomix@gmail.com',
      packages=['global_objectives'],
      install_requires=[
          'torch',
          'torchvision',
          'numpy',
          'sklearn',
      ],
      zip_safe=False)
