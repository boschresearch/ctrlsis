LAYER_LOOKUP = {

0 :'fc',    # ...................................................... torch.Size([4, 1024, 8, 8]) 

1 :'block_0_conv_on_shortcut_spade_norm_layer',    # ............... torch.Size([4, 1024, 8, 8]) 
2 :'block_0_first_spade_norm_layer_after_batchnorm',    # .......... torch.Size([4, 1024, 8, 8]) 
3 :'block_0_first_spade_norm_layer_output_of_mlp_shared',    # ..... torch.Size([4, 128, 8, 8]) 
4 :'block_0_first_spade_norm_layer_output_of_gamma',    # .......... torch.Size([4, 1024, 8, 8]) 
5 :'block_0_first_spade_norm_layer_output_of_beta',    # ........... torch.Size([4, 1024, 8, 8]) 
6 :'block_0_first_spade_norm_layer_output',    # ................... torch.Size([4, 1024, 8, 8]) 
7 :'block_0_conv_on_first_spade_norm_layer',    # .................. torch.Size([4, 1024, 8, 8]) 
8 :'block_0_second_spade_norm_layer_after_batchnorm',    # ......... torch.Size([4, 1024, 8, 8]) 
9 :'block_0_second_spade_norm_layer_output_of_mlp_shared',    # .... torch.Size([4, 128, 8, 8]) 
10 :'block_0_second_spade_norm_layer_output_of_gamma',   # ......... torch.Size([4, 1024, 8, 8]) 
11 :'block_0_second_spade_norm_layer_output_of_beta',   # .......... torch.Size([4, 1024, 8, 8]) 
12 :'block_0_second_spade_norm_layer_output',   # .................. torch.Size([4, 1024, 8, 8]) 
13 :'block_0_conv_on_second_spade_norm_layer',   # ................. torch.Size([4, 1024, 8, 8]) 
14 :'block_0_resnet_block_output',   # ............................. torch.Size([4, 1024, 8, 8]) 

15 :'block_1_conv_on_shortcut_spade_norm_layer',   # ............... torch.Size([4, 1024, 16, 16]) 
16 :'block_1_first_spade_norm_layer_after_batchnorm',   # .......... torch.Size([4, 1024, 16, 16]) 
17 :'block_1_first_spade_norm_layer_output_of_mlp_shared',   # ..... torch.Size([4, 128, 16, 16]) 
18 :'block_1_first_spade_norm_layer_output_of_gamma',   # .......... torch.Size([4, 1024, 16, 16]) 
19 :'block_1_first_spade_norm_layer_output_of_beta',   # ........... torch.Size([4, 1024, 16, 16]) 
20 :'block_1_first_spade_norm_layer_output',   # ................... torch.Size([4, 1024, 16, 16]) 
21 :'block_1_conv_on_first_spade_norm_layer',   # .................. torch.Size([4, 1024, 16, 16]) 
22 :'block_1_second_spade_norm_layer_after_batchnorm',   # ......... torch.Size([4, 1024, 16, 16]) 
23 :'block_1_second_spade_norm_layer_output_of_mlp_shared',   # .... torch.Size([4, 128, 16, 16]) 
24 :'block_1_second_spade_norm_layer_output_of_gamma',   # ......... torch.Size([4, 1024, 16, 16]) 
25 :'block_1_second_spade_norm_layer_output_of_beta',   # .......... torch.Size([4, 1024, 16, 16]) 
26 :'block_1_second_spade_norm_layer_output',   # .................. torch.Size([4, 1024, 16, 16]) 
27 :'block_1_conv_on_second_spade_norm_layer',   # ................. torch.Size([4, 1024, 16, 16]) 
28 :'block_1_resnet_block_output',   # ............................. torch.Size([4, 1024, 16, 16]) 

29 :'block_2_shortcut_spade_norm_layer_after_batchnorm',   # ....... torch.Size([4, 1024, 32, 32]) 
30 :'block_2_shortcut_spade_norm_layer_output_of_mlp_shared',   # .. torch.Size([4, 128, 32, 32]) 
31 :'block_2_shortcut_spade_norm_layer_output_of_gamma',   # ....... torch.Size([4, 1024, 32, 32]) 
32 :'block_2_shortcut_spade_norm_layer_output_of_beta',   # ........ torch.Size([4, 1024, 32, 32]) 
33 :'block_2_shortcut_spade_norm_layer_output',   # ................ torch.Size([4, 1024, 32, 32]) 
34 :'block_2_conv_on_shortcut_spade_norm_layer',   # ............... torch.Size([4, 512, 32, 32]) 
35 :'block_2_first_spade_norm_layer_after_batchnorm',   # .......... torch.Size([4, 1024, 32, 32]) 
36 :'block_2_first_spade_norm_layer_output_of_mlp_shared',   # ..... torch.Size([4, 128, 32, 32]) 
37 :'block_2_first_spade_norm_layer_output_of_gamma',   # .......... torch.Size([4, 1024, 32, 32]) 
38 :'block_2_first_spade_norm_layer_output_of_beta',   # ........... torch.Size([4, 1024, 32, 32]) 
39 :'block_2_first_spade_norm_layer_output',   # ................... torch.Size([4, 1024, 32, 32]) 
40 :'block_2_conv_on_first_spade_norm_layer',   # .................. torch.Size([4, 512, 32, 32]) 
41 :'block_2_second_spade_norm_layer_after_batchnorm',   # ......... torch.Size([4, 512, 32, 32]) 
42 :'block_2_second_spade_norm_layer_output_of_mlp_shared',   # .... torch.Size([4, 128, 32, 32]) 
43 :'block_2_second_spade_norm_layer_output_of_gamma',   # ......... torch.Size([4, 512, 32, 32]) 
44 :'block_2_second_spade_norm_layer_output_of_beta',   # .......... torch.Size([4, 512, 32, 32]) 
45 :'block_2_second_spade_norm_layer_output',   # .................. torch.Size([4, 512, 32, 32]) 
46 :'block_2_conv_on_second_spade_norm_layer',   # ................. torch.Size([4, 512, 32, 32]) 
47 :'block_2_resnet_block_output',   # ............................. torch.Size([4, 512, 32, 32]) 

48 :'block_3_shortcut_spade_norm_layer_after_batchnorm',   # ....... torch.Size([4, 512, 64, 64]) 
49 :'block_3_shortcut_spade_norm_layer_output_of_mlp_shared',   # .. torch.Size([4, 128, 64, 64]) 
50 :'block_3_shortcut_spade_norm_layer_output_of_gamma',   # ....... torch.Size([4, 512, 64, 64]) 
51 :'block_3_shortcut_spade_norm_layer_output_of_beta',   # ........ torch.Size([4, 512, 64, 64]) 
52 :'block_3_shortcut_spade_norm_layer_output',   # ................ torch.Size([4, 512, 64, 64]) 
53 :'block_3_conv_on_shortcut_spade_norm_layer',   # ............... torch.Size([4, 256, 64, 64]) 
54 :'block_3_first_spade_norm_layer_after_batchnorm',   # .......... torch.Size([4, 512, 64, 64]) 
55 :'block_3_first_spade_norm_layer_output_of_mlp_shared',   # ..... torch.Size([4, 128, 64, 64]) 
56 :'block_3_first_spade_norm_layer_output_of_gamma',   # .......... torch.Size([4, 512, 64, 64]) 
57 :'block_3_first_spade_norm_layer_output_of_beta',   # ........... torch.Size([4, 512, 64, 64]) 
58 :'block_3_first_spade_norm_layer_output',   # ................... torch.Size([4, 512, 64, 64]) 
59 :'block_3_conv_on_first_spade_norm_layer',   # .................. torch.Size([4, 256, 64, 64]) 
60 :'block_3_second_spade_norm_layer_after_batchnorm',   # ......... torch.Size([4, 256, 64, 64]) 
61 :'block_3_second_spade_norm_layer_output_of_mlp_shared',   # .... torch.Size([4, 128, 64, 64]) 
62 :'block_3_second_spade_norm_layer_output_of_gamma',   # ......... torch.Size([4, 256, 64, 64]) 
63 :'block_3_second_spade_norm_layer_output_of_beta',   # .......... torch.Size([4, 256, 64, 64]) 
64 :'block_3_second_spade_norm_layer_output',   # .................. torch.Size([4, 256, 64, 64]) 
65 :'block_3_conv_on_second_spade_norm_layer',   # ................. torch.Size([4, 256, 64, 64]) 
66 :'block_3_resnet_block_output',   # ............................. torch.Size([4, 256, 64, 64]) 

67 :'block_4_shortcut_spade_norm_layer_after_batchnorm',   # ....... torch.Size([4, 256, 128, 128]) 
68 :'block_4_shortcut_spade_norm_layer_output_of_mlp_shared',   # .. torch.Size([4, 128, 128, 128]) 
69 :'block_4_shortcut_spade_norm_layer_output_of_gamma',   # ....... torch.Size([4, 256, 128, 128]) 
70 :'block_4_shortcut_spade_norm_layer_output_of_beta',   # ........ torch.Size([4, 256, 128, 128]) 
71 :'block_4_shortcut_spade_norm_layer_output',   # ................ torch.Size([4, 256, 128, 128]) 
72 :'block_4_conv_on_shortcut_spade_norm_layer',   # ............... torch.Size([4, 128, 128, 128]) 
73 :'block_4_first_spade_norm_layer_after_batchnorm',   # .......... torch.Size([4, 256, 128, 128]) 
74 :'block_4_first_spade_norm_layer_output_of_mlp_shared',   # ..... torch.Size([4, 128, 128, 128]) 
75 :'block_4_first_spade_norm_layer_output_of_gamma',   # .......... torch.Size([4, 256, 128, 128]) 
76 :'block_4_first_spade_norm_layer_output_of_beta',   # ........... torch.Size([4, 256, 128, 128]) 
77 :'block_4_first_spade_norm_layer_output',   # ................... torch.Size([4, 256, 128, 128]) 
78 :'block_4_conv_on_first_spade_norm_layer',   # .................. torch.Size([4, 128, 128, 128]) 
79 :'block_4_second_spade_norm_layer_after_batchnorm',   # ......... torch.Size([4, 128, 128, 128]) 
80 :'block_4_second_spade_norm_layer_output_of_mlp_shared',   # .... torch.Size([4, 128, 128, 128]) 
81 :'block_4_second_spade_norm_layer_output_of_gamma',   # ......... torch.Size([4, 128, 128, 128]) 
82 :'block_4_second_spade_norm_layer_output_of_beta',   # .......... torch.Size([4, 128, 128, 128]) 
83 :'block_4_second_spade_norm_layer_output',   # .................. torch.Size([4, 128, 128, 128]) 
84 :'block_4_conv_on_second_spade_norm_layer',   # ................. torch.Size([4, 128, 128, 128]) 
85 :'block_4_resnet_block_output',   # ............................. torch.Size([4, 128, 128, 128]) 

86 :'block_5_shortcut_spade_norm_layer_after_batchnorm',   # ....... torch.Size([4, 128, 256, 256]) 
87 :'block_5_shortcut_spade_norm_layer_output_of_mlp_shared',   # .. torch.Size([4, 128, 256, 256]) 
88 :'block_5_shortcut_spade_norm_layer_output_of_gamma',   # ....... torch.Size([4, 128, 256, 256]) 
89 :'block_5_shortcut_spade_norm_layer_output_of_beta',   # ........ torch.Size([4, 128, 256, 256]) 
90 :'block_5_shortcut_spade_norm_layer_output',   # ................ torch.Size([4, 128, 256, 256]) 
91 :'block_5_conv_on_shortcut_spade_norm_layer',   # ............... torch.Size([4, 64, 256, 256]) 
92 :'block_5_first_spade_norm_layer_after_batchnorm',   # .......... torch.Size([4, 128, 256, 256]) 
93 :'block_5_first_spade_norm_layer_output_of_mlp_shared',   # ..... torch.Size([4, 128, 256, 256]) 
94 :'block_5_first_spade_norm_layer_output_of_gamma',   # .......... torch.Size([4, 128, 256, 256]) 
95 :'block_5_first_spade_norm_layer_output_of_beta',   # ........... torch.Size([4, 128, 256, 256]) 
96 :'block_5_first_spade_norm_layer_output',   # ................... torch.Size([4, 128, 256, 256]) 
97 :'block_5_conv_on_first_spade_norm_layer',   # .................. torch.Size([4, 64, 256, 256]) 
98 :'block_5_second_spade_norm_layer_after_batchnorm',   # ......... torch.Size([4, 64, 256, 256]) 
99 :'block_5_second_spade_norm_layer_output_of_mlp_shared',   # .... torch.Size([4, 128, 256, 256]) 
100 :'block_5_second_spade_norm_layer_output_of_gamma',  # ......... torch.Size([4, 64, 256, 256]) 
101 :'block_5_second_spade_norm_layer_output_of_beta',  # .......... torch.Size([4, 64, 256, 256]) 
102 :'block_5_second_spade_norm_layer_output',  # .................. torch.Size([4, 64, 256, 256]) 
103 :'block_5_conv_on_second_spade_norm_layer',  # ................. torch.Size([4, 64, 256, 256]) 
104 :'block_5_resnet_block_output',  # ............................. torch.Size([4, 64, 256, 256]) 
105 : 'x'  # ....................................................... torch.Size([4, 3, 256, 256])

}


































                                                                                                      



































                                                                                                         



































