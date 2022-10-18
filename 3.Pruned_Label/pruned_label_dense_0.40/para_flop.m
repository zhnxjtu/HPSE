clc
clear all
close all

for i = 1:24
    name = ['./FM' num2str(i) '.txt'];
    a = load(name);
    l(i) = length(a);
end

l(25:36) = 0;

s(1) = l(1);
for j = 2:36
    s(j) = s(j-1) + l(j);
end

p_ori = (3*24*3*3 + ...
    24*12*3*3 + 36*12*3*3 + 48*12*3*3 + 60*12*3*3 + 72*12*3*3 + 84*12*3*3 + 96*12*3*3 + 108*12*3*3 + 120*12*3*3 + 132*12*3*3 + ...
    144*12*3*3 + 156*12*3*3 + 168*168*1*1 + ...
    168*12*3*3 + 180*12*3*3 + 192*12*3*3 + 204*12*3*3 + 216*12*3*3 + 228*12*3*3 + 240*12*3*3 + 252*12*3*3 + 264*12*3*3 + 276*12*3*3 + ...
    288*12*3*3 + 300*12*3*3 + 312*312*1*1 + ...
    312*12*3*3 + 324*12*3*3 + 336*12*3*3 + 348*12*3*3 + 360*12*3*3 + 372*12*3*3 + 384*12*3*3 + 396*12*3*3 + 408*12*3*3 + 420*12*3*3 + ...
    432*12*3*3 + 444*12*3*3 + ... 
    456*10)/10^6;
    
p_cur = (3*24*3*3 + ...
    24*(12-l(1))*3*3 + (36-s(1))*(12-l(2))*3*3 + (48-s(2))*(12-l(3))*3*3 + (60-s(3))*(12-l(4))*3*3 + (72-s(4))*(12-l(5))*3*3 + ...
    (84-s(5))*(12-l(6))*3*3 + (96-s(6))*(12-l(7))*3*3 + (108-s(7))*(12-l(8))*3*3 + (120-s(8))*(12-l(9))*3*3 + (132-s(9))*(12-l(10))*3*3 + ...
    (144-s(10))*(12-l(11))*3*3 + (156-s(11))*(12-l(12))*3*3 + (168-s(12))*(168-s(12))*1*1 + ...
    (168-s(12))*(12-l(13))*3*3 + (180-s(13))*(12-l(14))*3*3 + (192-s(14))*(12-l(15))*3*3 + (204-s(15))*(12-l(16))*3*3 + (216-s(16))*(12-l(17))*3*3 + ...
    (228-s(17))*(12-l(18))*3*3 + (240-s(18))*(12-l(19))*3*3 + (252-s(19))*(12-l(20))*3*3 + (264-s(20))*(12-l(21))*3*3 + ...
    (276-s(21))*(12-l(22))*3*3 + (288-s(22))*(12-l(23))*3*3 + (300-s(23))*(12-l(24))*3*3 + ...
    (312-s(24))*(312-s(24))*1*1 + ...
    (312-s(24))*(12-l(25))*3*3 + (324-s(25))*(12-l(26))*3*3 + (336-s(26))*(12-l(27))*3*3 + (348-s(27))*(12-l(28))*3*3 + (360-s(28))*(12-l(29))*3*3 + ...
    (372-s(29))*(12-l(30))*3*3 + (384-s(30))*(12-l(31))*3*3 + (396-s(31))*(12-l(32))*3*3 + (408-s(32))*(12-l(33))*3*3 + ...
    (420-s(33))*(12-l(34))*3*3 + (432-s(34))*(12-l(35))*3*3 + (444-s(35))*(12-l(36))*3*3 + ... 
    (456-s(36))*10)/10^6;

(p_ori - p_cur)/p_ori

f_ori = ((3*24*3*3 + ...
    24*12*3*3 + 36*12*3*3 + 48*12*3*3 + 60*12*3*3 + 72*12*3*3 + 84*12*3*3 + 96*12*3*3 + 108*12*3*3 + 120*12*3*3 + 132*12*3*3 + ...
    144*12*3*3 + 156*12*3*3 + 168*168*1*1)*32*32 + ...
    (168*12*3*3 + 180*12*3*3 + 192*12*3*3 + 204*12*3*3 + 216*12*3*3 + 228*12*3*3 + 240*12*3*3 + 252*12*3*3 + 264*12*3*3 + 276*12*3*3 + ...
    288*12*3*3 + 300*12*3*3 + 312*312*1*1)*16*16 + ...
    (312*12*3*3 + 324*12*3*3 + 336*12*3*3 + 348*12*3*3 + 360*12*3*3 + 372*12*3*3 + 384*12*3*3 + 396*12*3*3 + 408*12*3*3 + 420*12*3*3 + ...
    432*12*3*3 + 444*12*3*3)*8*8 + ... 
    456*10)/10^6;

f_cur = ((3*24*3*3 + ...
    24*(12-l(1))*3*3 + (36-s(1))*(12-l(2))*3*3 + (48-s(2))*(12-l(3))*3*3 + (60-s(3))*(12-l(4))*3*3 + (72-s(4))*(12-l(5))*3*3 + ...
    (84-s(5))*(12-l(6))*3*3 + (96-s(6))*(12-l(7))*3*3 + (108-s(7))*(12-l(8))*3*3 + (120-s(8))*(12-l(9))*3*3 + (132-s(9))*(12-l(10))*3*3 + ...
    (144-s(10))*(12-l(11))*3*3 + (156-s(11))*(12-l(12))*3*3 + (168-s(12))*(168-s(12))*1*1)*32*32 + ...
    ((168-s(12))*(12-l(13))*3*3 + (180-s(13))*(12-l(14))*3*3 + (192-s(14))*(12-l(15))*3*3 + (204-s(15))*(12-l(16))*3*3 + (216-s(16))*(12-l(17))*3*3 + ...
    (228-s(17))*(12-l(18))*3*3 + (240-s(18))*(12-l(19))*3*3 + (252-s(19))*(12-l(20))*3*3 + (264-s(20))*(12-l(21))*3*3 + ...
    (276-s(21))*(12-l(22))*3*3 + (288-s(22))*(12-l(23))*3*3 + (300-s(23))*(12-l(24))*3*3 + ...
    (312-s(24))*(312-s(24))*1*1)*16*16 + ...
    ((312-s(24))*(12-l(25))*3*3 + (324-s(25))*(12-l(26))*3*3 + (336-s(26))*(12-l(27))*3*3 + (348-s(27))*(12-l(28))*3*3 + (360-s(28))*(12-l(29))*3*3 + ...
    (372-s(29))*(12-l(30))*3*3 + (384-s(30))*(12-l(31))*3*3 + (396-s(31))*(12-l(32))*3*3 + (408-s(32))*(12-l(33))*3*3 + ...
    (420-s(33))*(12-l(34))*3*3 + (432-s(34))*(12-l(35))*3*3 + (444-s(35))*(12-l(36))*3*3)*8*8 + ... 
    (456-s(36))*10)/10^6;

(f_ori - f_cur)/f_ori