#import "@preview/humble-dtu-thesis:0.1.0": *

= Introduction

What exactly, in simple terms, is VV, VH, HH, and HV

They’re just labels for how the radar wave is polarized when it’s sent out and when it’s received.

Think of polarization as the “direction the wave is wiggling”:

H = Horizontal

V = Vertical

Sentinel-1 can transmit a wave (first letter) and then listen to the echo (second letter).

So:

VV: transmit Vertical, receive Vertical

VH: transmit Vertical, receive Horizontal

HH: transmit Horizontal, receive Horizontal

HV: transmit Horizontal, receive Vertical

Co-pol vs cross-pol (the useful intuition)

VV and HH are co-polarized (same on transmit & receive)

usually stronger signal

often highlights surface roughness and built-up areas well

VH and HV are cross-polarized (it “flipped” polarization)

usually weaker signal

often responds more to vegetation / volume scattering (leaves/branches randomize the wave)

What Sentinel-1 typically provides

Most Sentinel-1 GRD products are either:

VV + VH (called DV = dual-V), or

HH + HV (called DH = dual-H)

So you’ll usually see two images per product: one co-pol (VV or HH) and one cross-pol (VH or HV).


For the NN, use VV, VH, and (VV - VH) / (VV + VH) as inputs

- Why is this interesting @WikipediaBanana

#lorem(250)

// #add-note[Remember to fact check this]
