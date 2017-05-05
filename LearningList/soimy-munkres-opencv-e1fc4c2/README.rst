
munkres-opencv
===========

An implementation of the Kuhn–Munkres algorithm forked from John Weaver
Simplified the source code structure and ported to support OPENCV only.
cv::Mat_<int> has been used instead of John's matrix class. It is cleaner and better support for empty matrix cases in OPENCV multiple object tracking.


Usage
-----

You have to have OPENCV 2.4 or above installed to use this.
For MacOS user, Xcode project file is included.
Basic usage can be referred from main.cpp 

License
-------

Port to OPENCV by Shen Yiming 2014

Copyright (c) 2007-2013 John Weaver and contributors.

Licensed under the GPLv2. See the file COPYING for details.


