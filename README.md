# Refactored-Guac (Working title)

## What is it?

This will likely become a Python flask based API that will be used for training ML models. The goal will be to run this on a Docker container running nvidia-docker, with the eventuality of being ported over to Kubernetes. 

## The goal for now

Convert all of the training performed in Jeremy Howard's fast.ai [course1](http://course.fast.ai/index.html) notebooks to API calls to this service rather than the local python environment.

## What to do first

Will start off by learning the basics of Flask and how to build an API on it. Once I get that working, next steps will be to specify the scope of the API. 

## Goals to implement

- OAuth2 workflow
- Read training data from external sources (GCS, S3, etc)
- Dashboard
- REST API (Actual goal)
