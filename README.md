

# SNN simulator 

## Description
Spiking neural network simulator based on memristor characteristics


## Table of Contents 

- [Installation](#installation)
- [Theory](#Theory)
- [Build_network](#Build_network)
- [Credits](#credits)
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



## Build_network
<p align="center">
  <img src="images/stag.png?raw=true" 
       width="400" 
       height="580"/>
</p>
## Credits

None

## License

## How to Contribute

 [Contributor Covenant](https://www.contributor-covenant.org/) 

## Tests


