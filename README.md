# cudaRenderer

Simple renderer running on CUDA. returns output to simple r,g,b format file. Write own drawing tool or use the one provided on my other repo(pygame-renderer)

![image](https://github.com/user-attachments/assets/d6147ef5-c53e-48ae-8775-2ff6a4be5d2b)

example terminal output of a successful run


TODO:

12/24: 
- Fix reflection math and associated bugs
- Reduce register usage to increase num concurrent warps running
- Correctly implement --cuda_fast_math to use more efficient math instructions(__fsqrtf, __fdivf_)
- Start setting up simple model loader(STL files) with own custom material files

1/25 - 3/25:
- Work on BVH
- Volumetrics?
- Add automatic profiler instead of relying on predefined macros(more user-friendly and cross-platform stuff)
