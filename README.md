# Interior-Point Differential Dynamic Programming (IP-DDP) and iterative Linear Quadratic Regulator (IP-iLQR)

This repository provides a Python reimplementations of the Interor-Point DDP and iLQR [1]. 
You can find the original code repository [here](https://github.com/xapavlov/ipddp).
This implementation includes some minor modifications to the original version:
- The ability to select initial state elements as optimization variables.
- Support for state box constraints at the terminal time step.

This implementation does not support starting from an infeasible initial solution.

## Installation
-------------
To use this repository, simply clone it. You'll need to install the Numpy package using:
```
pip install numpy
```

## Examples
To run the car trajectory planning example, use the following command:
```
./scripts/run_car_traj_plan.bash
```

The animation below shows the optimization progress. 
The left panel displays the xy-path, while the panels on the right show the time series of states and control inputs. 
 

## Citations
[1] A. Pavlov, I. Shames and C. Manzie, "Interior Point Differential Dynamic Programming," in IEEE Transactions on Control Systems Technology, vol. 29, no. 6, pp. 2720-2727, Nov. 2021.  




