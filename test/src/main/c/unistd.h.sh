#!/bin/bash

jextract --output ../java -t demo.ffm.unistd --header-class-name Unistd_h  /usr/include/unistd.h
# --output 输出路径（不包含包名）
# -t 生成的包名
# --header-class-name 生成的头文件对应的Java类名
# -l 库名称，使用“:”开头则为库路径
# --use-system-load-library 使用 System::load 方法加载 -l 指定的库

