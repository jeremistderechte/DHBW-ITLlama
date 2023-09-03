# DHBW-ITLlama - IT Chabot for DHBW

Welcome to the DHBW-ITLlama project! This repository contains everything you need to fine-tune our Llama2 based IT Chatbot for the DHBW (Baden-Wuerttemberg Cooperative State University).

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Usage](#usage)



## Introduction

Llama 2 is a state-of-the-art natural language processing model designed to understand and generate text in a wide range of languages. This repository provides tools and resources to fine-tune the LLM model to function as an IT Chatbot.

## Features

- A lot things to do

## Getting Started

### Prerequisites

Before you start, ensure you have the following prerequisites:

- Cuda compatible GPU
  - Minimum Cuda compability 5.0 (Maxwell), Ampere or newer recommended (fast 16bit Training)
  - Kepler (3.5, 3.7) only partily compatible, if you compile PyTorch etc. for yourself (sorry K20, K40, K80 Owners :-()

- You GPU needs at least 12GB of VRAM for the smallest 7B Modell to finetune (for inference with 4bit precision, you need less)
  - Google Colab offers free Tesla T4 (Turing) with 16 GB (15GB usable) of VRAM for that purpose


### Usage

Clone the repository and run the notebooks:

```bash
git clone https://github.com/jeremistderechte/DHBW-ITLlama.git
cd DHBW-ITLlama
jupyter lab //alternatively google colab
