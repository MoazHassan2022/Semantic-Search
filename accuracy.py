actual_ids = [275, 114, 699, 552, 344, 346, 519, 970, 967, 130, 667, 579, 747, 985, 588, 622, 496, 478, 298, 864, 334, 217, 724, 129, 54, 482, 196, 38, 379, 286]
potential_ids = [0, 1, 4, 6, 7, 9, 11, 12, 13, 15, 16, 18, 22, 23, 24, 27, 29, 30, 31, 36, 37, 38, 39, 41, 43, 44, 51, 52, 54, 58, 59, 62, 65, 66, 68, 69, 72, 76, 77, 78, 81, 85, 88, 89, 90, 
92, 93, 94, 95, 96, 97, 99, 101, 102, 104, 105, 108, 109, 110, 111, 113, 114, 115, 118, 120, 124, 126, 128, 129, 130, 132, 135, 137, 139, 143, 144, 146, 147, 148, 149, 150, 154, 156, 158, 159, 160, 161, 164, 167, 173, 174, 179, 180, 183, 185, 191, 192, 193, 194, 197, 199, 202, 211, 213, 217, 218, 219, 220, 222, 223, 227, 229, 230, 232, 234, 237, 239, 243, 247, 248, 252, 253, 255, 256, 258, 261, 262, 264, 265, 266, 268, 269, 272, 273, 274, 275, 276, 278, 280, 281, 282, 283, 284, 286, 287, 288, 290, 293, 294, 295, 296, 298, 300, 301, 302, 304, 306, 307, 310, 317, 318, 319, 320, 321, 323, 324, 325, 326, 329, 331, 333, 335, 337, 338, 339, 340, 342, 344, 345, 346, 348, 349, 350, 351, 353, 354, 355, 359, 360, 361, 362, 363, 364, 367, 370, 371, 377, 379, 387, 390, 393, 395, 397, 398, 400, 402, 411, 416, 418, 420, 423, 426, 427, 429, 431, 437, 438, 441, 442, 443, 444, 445, 448, 450, 451, 452, 453, 454, 455, 459, 460, 461, 463, 464, 467, 476, 479, 481, 482, 485, 489, 490, 496, 498, 499, 504, 508, 509, 510, 511, 512, 513, 515, 518, 520, 522, 526, 530, 531, 533, 535, 537, 538, 540, 541, 543, 545, 550, 552, 554, 555, 557, 561, 562, 564, 565, 566, 569, 572, 575, 576, 577, 578, 579, 580, 581, 583, 586, 588, 589, 590, 595, 597, 603, 605, 607, 608, 609, 610, 611, 614, 615, 616, 618, 620, 621, 622, 625, 629, 630, 631, 632, 634, 637, 638, 641, 643, 644, 645, 649, 650, 651, 652, 654, 657, 658, 661, 667, 668, 669, 674, 676, 677, 678, 679, 683, 688, 691, 692, 693, 695, 696, 699, 700, 701, 702, 707, 714, 715, 716, 720, 721, 722, 723, 724, 727, 729, 730, 732, 735, 736, 737, 739, 740, 741, 742, 745, 747, 748, 749, 750, 751, 752, 754, 755, 759, 760, 761, 763, 764, 768, 769, 773, 774, 780, 781, 790, 794, 797, 798, 799, 800, 802, 803, 807, 808, 811, 812, 813, 814, 815, 816, 817, 819, 820, 821, 822, 824, 825, 826, 827, 830, 831, 832, 834, 836, 839, 840, 841, 845, 851, 853, 858, 859, 860, 861, 863, 864, 866, 867, 869, 870, 873, 875, 877, 885, 888, 892, 896, 897, 899, 901, 902, 903, 906, 907, 908, 909, 910, 911, 915, 920, 921, 922, 925, 927, 928, 929, 930, 931, 932, 934, 935, 936, 938, 939, 941, 942, 943, 946, 947, 948, 949, 950, 953, 955, 959, 961, 963, 965, 966, 968, 969, 970, 971, 973, 975, 977, 979, 981, 982, 988, 991, 992, 997]
c = 0
for actual_id in actual_ids:
    if actual_id not in potential_ids:
        c += 1


print(c)