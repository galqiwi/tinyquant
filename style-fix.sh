#!/bin/bash

set -e

uvx ruff format

uvx ruff check --fix

uvx ruff check --select I --fix
