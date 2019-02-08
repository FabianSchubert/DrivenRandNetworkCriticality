#!/bin/bash

grep -o "cite{.*}" $1 | grep ".*," -o
