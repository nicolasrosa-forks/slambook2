clc
clear all

addpath('quaternions/')

%% Quaternion order
syms q1 q2 q3 q4
q = [q1, q2, q3, q4]  % q = q4 + q1*i + q2*j + q3*k

% Symbolic example
syms sa xa ya za
syms sb xb yb zb
qmult([xa, ya, za, sa], [xb, yb, zb, sb])

% Numeric Example
qmult([1,2,3,4], [1,2,3,4])  % out = [8, 16, 24, 2],  q = 2 + 8*i + 16*j + 24*k


%% q_p' = q_rot*q_p*inv(q_rot)
syms p1 p2 p3
syms v1 v2 v3 s

q_p = [p1, p2, p3, 0]
q_rot = [v1, v2, v3, s]
q_rot_inv = [-v1, -v2, -v3, s]

q_p_new = qmult(qmult(q_rot, q_p), q_rot_inv)
q_p_new = expand(q_p_new)