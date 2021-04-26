from mmgen.models.architectures import LSGANDiscriminator, LSGANGenerator

g = LSGANGenerator()
x = g(None, 10)
print(x.shape)
d = LSGANDiscriminator()
y = d(x)
print(y.shape)
