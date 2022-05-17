from setuptools import setup, find_packages

setup(
  name = 'enformer-jax',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Enformer - Attention network for predicting gene expression, in Jax',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/enformer-jax',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'gene expression'
  ],
  install_requires=[
    'dm-haiku>=0.0.4',
    'einops==0.4',
    'jax>=0.3.4',
    'jaxlib>=0.1',
    'jmp>=0.0.2',
    'optax>=0.0.9',
    'numpy'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
