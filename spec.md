### Physical Learning Machine Design Spec

#### Data Structure
#### Resistors
* Directly store in PySpice circuit object
#### Differential voltage sources
* Maintain 2 lists: inputs and outputs
* Each list has dimension $2 \times n$ in form $[v_i^-, v_i^+]$
#### Constructor
* Input: (directed) connectivity graph, node configuration arrays
* Graph format: 
    * linear resistive edges have `type` property of "resistor" and `weight` property representing edge resistance
* Construct PySpice circuit where each node in the graph represents a node in the SPICE model
* Resistive edge corresponding to each edge, with type initialized to resistance corresponding to edge's weight
* "ReLu" edges contain a resistor and diode in series, as described by input connectivity graph
* Construct array of input sources and output sources, according to node config 
* Initialize all sources to disabled
* Connect resistors/nonlinear edges according to connectivity matrix
* Throw away connectivity matrix 
#### Evaluation
* Inputs: n-length list of differential input clamps, (optional) n-length list of differential output clamps
* Enable all inputs and set to values according to inputs
* If outputs specified, enable and set values, else disable
* Evaluate using SPICE and return all absolute node voltages (or only output differential voltages)
#### Training
* Inputs: initialized network, mxn length input array, mxn' output array  
For every epoch/data row:
1. Evaluate free state on inputs only
2. Compute nudges: $V_c = \eta \cdot y + (1-\eta) V_f$
3. Evaluate clamp state with both input and output nudges
4. Compute power dissipated through resistors using Numpy methods
5. Create updates w.r.t. either admittance or resistance
6. For each resistor update based on change in admittance

### Addenum: Nonlinear Networks
#### Constructor
* Same as above, but input graph can optionally have tunable diode edges
* nonlinear "ReLu" edges have a `type` property of "diode", a direction corresponding to the direction of the diode, and a `weight` property representing edge resistance
* Diode edges have type "diode" specified, resistors have edge "resistor" specified, all other edges ignored
* Filter for edges with "resistor" type and pass to parent constructor as above.