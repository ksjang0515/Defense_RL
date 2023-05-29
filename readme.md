Game info

map_size = (256, 256)

TODO

flax framework

1. check whether setup method could be used when using nn.compact

2. check how recurrent model is used in flax. in other frameworks 2nd dimension is used as timestep, but it seems like in flax the programmer needs to manually run a loop to do so. current implementation is based on the latter.
