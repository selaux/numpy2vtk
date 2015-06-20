#!/usr/bin/env bash

coverage run -m unittest discover -s test -p '*_test.py'
coverage html
coverage report -m
