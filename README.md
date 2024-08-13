

# Memristor crossbar & SNN simulator 

## Description
Spiking neural network simulator based on memristor characteristics and crossbar architecture


## Table of Contents 

- [Installation](#installation)
- [Theory](#Theory)
- [Build_network](#Build-network)
- [Crossbar architecture](#Crossbar-architecture)
- [License](#license)

## Installation
If you whant to run some examples or construct your own SNN clone repo
```
git clone https://github.com/anddudkin/anddudkin_mem_project.git
```
## Theory
Spiking neural network (SNN) operates with spikes. SNN takes spikes as input and produce output spikes with respect to learning rule.

<p align="center">
  <img src="images/snn.png?raw=true"
       width="400" 
       height="200"/>
</p>

We can use different types of neurons (IF,LIF, etc.). Dinamics of Leaky Integrate and Fire (LIF) neuron membrance presented in the picture below.

<p align="center">
  <img src="images/lif.png?raw=true"
       width="350" 
       height="280"/>
</p>

Implemented one of the basic SNN learning rules - STDP (Spike-timing dependent plasticity)

<p align="center">
  <img src="images/stdp.png?raw=true" 
       width="350" 
       height="280"/>
</p>

Weights plot from MNIST_example.py

<p align="center">
  <img src="images/example_mnist.png?raw=true" 
       width="350" 
       height="350"/>
</p>

## Build_network
Baseline for constructing your own network
<p align="center">
  <img src="images/stag.png?raw=true" 
       width="400" 
       height="580"/>
</p>

## Crossbar architecture
Implemented crossbar takes into account impact of wire resistance
<p align="center">
  <img src="images/crossbar.png?raw=true" 
       width="450" 
       height="450"/>
</p>
Also descrete conductance states were extracted from real memristive device and used in following simulations
<p align="center">
  <img src="images/stages.jpg?raw=true" 
       width="500" 
       height="400"/>
</p>

## License

## How to Contribute

 [Contributor Covenant](https://www.contributor-covenant.org/) 




