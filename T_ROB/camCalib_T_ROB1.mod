MODULE camCalib_T_ROB1
VAR extjoint extj := [9E9,9E9,9E9,9E9,9E9,9E9];
VAR confdata conf := [0,0,0,0];
PERS tooldata kinectAzure:=[TRUE,[[50.721,-32.000,77.725],[-0.50000,0.50000,-0.50000,0.50000]],[2.000,[50.721,-32.000,77.725],[1,0,0,0],0,0,0]];
TASK PERS wobjdata DefaultFrame:=[FALSE,TRUE,"",[[0.000,0.000,0.000],[1.00000,0.00000,0.00000,0.00000]],[[0,0,0],[1,0,0,0]]];
TASK PERS speeddata Speed000:=[500.000,180.000,5000.000,1080.000];
TASK PERS speeddata Speed001:=[250.000,180.000,5000.000,1080.000];
TASK PERS zonedata Zone000:=[FALSE,1.000,1.000,1.000,0.100,1.000,0.100];
TASK PERS zonedata Zone001:=[FALSE,0.010,0.010,0.010,0.001,0.010,0.001];
PROC Main()
ConfL \Off;
PathAccLim FALSE, TRUE\DecelMax := .1;
SetDO \Sync, DO_0, 0;
SetDO \Sync, DO_1, 0;
MoveAbsJ [[1,0,-1,1,-1,1],extj],Speed000,Zone000,kinectAzure;
SetDO \Sync, DO_0, 1;
MoveL [[779.534,566.312,922.398],[-0.1325,-0.51546,0.69458,-0.48405],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[721.262,468.391,1014.221],[-0.05917,-0.56323,0.70463,-0.42752],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[675.172,350,1086.847],[0.01481,-0.60483,0.70695,-0.3663],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[643.279,216.312,1137.103],[0.08862,-0.63981,0.70153,-0.30107],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[626.976,73.17,1162.792],[0.16147,-0.66777,0.68842,-0.23254],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[626.976,-73.17,1162.792],[-0.23254,0.68842,-0.66777,0.16147],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[643.279,-216.312,1137.103],[-0.30107,0.70153,-0.63981,0.08862],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[675.172,-350,1086.847],[-0.3663,0.70695,-0.60483,0.01481],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[721.262,-468.391,1014.221],[-0.42752,0.70463,-0.56323,-0.05917],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[779.534,-566.312,922.398],[-0.48405,0.69458,-0.51546,-0.1325],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[817.055,-566.312,943.54],[-0.45641,0.6867,-0.54009,-0.16867],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[768.7,-468.391,1040.95],[-0.39745,0.70056,-0.58483,-0.09597],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[730.454,-350,1117.996],[-0.33414,0.70676,-0.62318,-0.02221],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[703.989,-216.312,1171.311],[-0.26717,0.70521,-0.65469,0.05179],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[690.46,-73.17,1198.563],[-0.19728,0.69593,-0.67903,0.12522],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[690.46,73.17,1198.563],[0.12522,-0.67903,0.69593,-0.19728],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[703.989,216.312,1171.311],[0.05179,-0.65469,0.70521,-0.26717],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[730.454,350,1117.996],[-0.02221,-0.62318,0.70676,-0.33414],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[768.7,468.391,1040.95],[-0.09597,-0.58483,0.70056,-0.39745],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[817.055,566.312,943.54],[-0.16867,-0.54009,0.6867,-0.45641],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[856.58,566.312,960.644],[-0.20438,-0.56323,0.67693,-0.42752],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[818.672,468.391,1062.575],[-0.1325,-0.60483,0.69458,-0.3663],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[788.689,350,1143.197],[-0.05917,-0.63981,0.70463,-0.30107],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[767.942,216.312,1198.986],[0.01481,-0.66777,0.70695,-0.23254],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[757.336,73.17,1227.503],[0.08862,-0.68842,0.70153,-0.16147],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[757.336,-73.17,1227.503],[-0.16147,0.70153,-0.68842,0.08862],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[767.942,-216.312,1198.986],[-0.23254,0.70695,-0.66777,0.01481],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[788.689,-350,1143.197],[-0.30107,0.70463,-0.63981,-0.05917],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[818.672,-468.391,1062.575],[-0.3663,0.69458,-0.60483,-0.1325],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[856.58,-566.312,960.644],[-0.42752,0.67693,-0.56323,-0.20438],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[897.677,-566.312,973.523],[-0.39745,0.6653,-0.58483,-0.23952],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[870.631,-468.391,1078.858],[-0.33414,0.6867,-0.62318,-0.16867],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[849.24,-350,1162.172],[-0.26717,0.70056,-0.65469,-0.09597],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[834.437,-216.312,1219.824],[-0.19728,0.70676,-0.67903,-0.02221],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[826.871,-73.17,1249.294],[-0.12522,0.70521,-0.69593,0.05179],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[826.871,73.17,1249.294],[0.05179,-0.69593,0.70521,-0.12522],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[834.437,216.312,1219.824],[-0.02221,-0.67903,0.70676,-0.19728],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[849.24,350,1162.172],[-0.09597,-0.65469,0.70056,-0.26717],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[870.631,468.391,1078.858],[-0.16867,-0.62318,0.6867,-0.33414],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[897.677,566.312,973.523],[-0.23952,-0.58483,0.6653,-0.39745],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[939.894,566.312,982.036],[-0.27401,-0.60483,0.65186,-0.3663],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[924.007,468.391,1089.621],[-0.20438,-0.63981,0.67693,-0.30107],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[911.442,350,1174.714],[-0.1325,-0.66777,0.69458,-0.23254],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[902.747,216.312,1233.598],[-0.05917,-0.68842,0.70463,-0.16147],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[898.302,73.17,1263.697],[0.01481,-0.70153,0.70695,-0.08862],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[898.302,-73.17,1263.697],[-0.08862,0.70695,-0.70153,0.01481],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[902.747,-216.312,1233.598],[-0.16147,0.70463,-0.68842,-0.05917],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[911.442,-350,1174.714],[-0.23254,0.69458,-0.66777,-0.1325],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[924.007,-468.391,1089.621],[-0.30107,0.67693,-0.63981,-0.20438],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[939.894,-566.312,982.036],[-0.3663,0.65186,-0.60483,-0.27401],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[982.77,-566.312,986.089],[-0.33414,0.63662,-0.62318,-0.30775],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[978.216,-468.391,1094.745],[-0.26717,0.6653,-0.65469,-0.23952],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[974.614,-350,1180.686],[-0.19728,0.6867,-0.67903,-0.16867],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[972.122,-216.312,1240.156],[-0.12522,0.70056,-0.69593,-0.09597],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[970.848,-73.17,1270.555],[-0.05179,0.70676,-0.70521,-0.02221],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[970.848,73.17,1270.555],[-0.02221,-0.70521,0.70676,-0.05179],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[972.122,216.312,1240.156],[-0.09597,-0.69593,0.70056,-0.12522],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[974.614,350,1180.686],[-0.16867,-0.67903,0.6867,-0.19728],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[978.216,468.391,1094.745],[-0.23952,-0.65469,0.6653,-0.26717],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[982.77,566.312,986.089],[-0.30775,-0.62318,0.63662,-0.33414],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1025.835,566.312,985.638],[0.34065,0.63981,-0.61964,0.30107],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1032.664,468.391,1094.175],[0.27401,0.66777,-0.65186,0.23254],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1038.065,350,1180.022],[0.20438,0.68842,-0.67693,0.16147],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1041.802,216.312,1239.426],[0.1325,0.70153,-0.69458,0.08862],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1043.713,73.17,1269.792],[0.05917,0.70695,-0.70463,0.01481],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1043.713,-73.17,1269.792],[0.01481,-0.70463,0.70695,0.05917],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1041.802,-216.312,1239.426],[0.08862,-0.69458,0.70153,0.1325],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1038.065,-350,1180.022],[0.16147,-0.67693,0.68842,0.20438],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1032.664,-468.391,1094.175],[0.23254,-0.65186,0.66777,0.27401],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1025.835,-566.312,985.638],[0.30107,-0.61964,0.63981,0.34065],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1068.617,-566.312,980.688],[0.26717,-0.60096,0.65469,0.37261],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1086.753,-468.391,1087.916],[0.19728,-0.63662,0.67903,0.30775],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1101.098,-350,1172.728],[0.12522,-0.6653,0.69593,0.23952],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1111.025,-216.312,1231.417],[0.05179,-0.6867,0.70521,0.16867],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1116.099,-73.17,1261.416],[-0.02221,-0.70056,0.70676,0.09597],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1116.099,73.17,1261.416],[0.09597,0.70676,-0.70056,-0.02221],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1111.025,216.312,1231.417],[0.16867,0.70521,-0.6867,0.05179],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1101.098,350,1172.728],[0.23952,0.69593,-0.6653,0.12522],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1086.753,468.391,1087.916],[0.30775,0.67903,-0.63662,0.19728],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1068.617,566.312,980.688],[0.37261,0.65469,-0.60096,0.26717],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1110.647,566.312,971.293],[0.40356,0.66777,-0.58064,0.23254],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1139.892,468.391,1076.038],[0.34065,0.68842,-0.61964,0.16147],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1163.024,350,1158.886],[0.27401,0.70153,-0.65186,0.08862],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1179.031,216.312,1216.215],[0.20438,0.70695,-0.67693,0.01481],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1187.213,73.17,1245.52],[0.1325,0.70463,-0.69458,-0.05917],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1187.213,-73.17,1245.52],[-0.05917,-0.69458,0.70463,0.1325],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1179.031,-216.312,1216.215],[0.01481,-0.67693,0.70695,0.20438],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1163.024,-350,1158.886],[0.08862,-0.65186,0.70153,0.27401],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1139.892,-468.391,1076.038],[0.16147,-0.61964,0.68842,0.34065],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1110.647,-566.312,971.293],[0.23254,-0.58064,0.66777,0.40356],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1151.465,-566.312,957.556],[0.19728,-0.55872,0.67903,0.43339],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1191.499,-468.391,1058.671],[0.12522,-0.60096,0.69593,0.37261],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1223.164,-350,1138.647],[0.05179,-0.63662,0.70521,0.30775],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1245.075,-216.312,1193.989],[-0.02221,-0.6653,0.70676,0.23952],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1256.276,-73.17,1222.278],[-0.09597,-0.6867,0.70056,0.16867],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1256.276,73.17,1222.278],[0.16867,0.70056,-0.6867,-0.09597],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1245.075,216.312,1193.989],[0.23952,0.70676,-0.6653,-0.02221],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1223.164,350,1138.647],[0.30775,0.70521,-0.63662,0.05179],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1191.499,468.391,1058.671],[0.37261,0.69593,-0.60096,0.12522],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveL [[1151.465,566.312,957.556],[0.43339,0.67903,-0.55872,0.19728],conf,extj],Speed001,Zone001,kinectAzure \WObj:=DefaultFrame;
WaitTime \InPos, 2;
SetDO \Sync, DO_1, 1;
WaitTime 0.5;
SetDO \Sync, DO_1, 0;
MoveAbsJ [[1,0,-1,1,-1,1],extj],Speed000,Zone000,kinectAzure;
SetDO \Sync, DO_0, 0;
ENDPROC
ENDMODULE