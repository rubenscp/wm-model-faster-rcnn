#!/bin/bash
echo $1
dos2unix $1 
qsub $1
