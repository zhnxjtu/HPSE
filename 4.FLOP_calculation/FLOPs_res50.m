function [p_per, f_per] = FLOPs_res50(name_flod)

    for i = 1:48
        name = [name_flod '/FM' num2str(i) '.txt'];
        a = load(name);
        l(i) = length(a);
    end

    p_ori = (3*64*7*7 + ...
        64*64*1*1 + 64*64*3*3 + 64*256*1*1 + 64*256*1*1 + 256*64*1*1 + 64*64*3*3 + 64*256*1*1 + 256*64*1*1 + 64*64*3*3 + 64*256*1*1 + ...
        256*128*1*1 + 128*128*3*3 + 128*512*1*1 + 256*512*1*1 + 512*128*1*1 + 128*128*3*3 + 128*512*1*1 + ...
        512*128*1*1 + 128*128*3*3 + 128*512*1*1 + 512*128*1*1 + 128*128*3*3 + 128*512*1*1 + ...
        512*256*1*1 + 256*256*3*3 + 256*1024*1*1 + 512*1024*1*1 + 1024*256*1*1 + 256*256*3*3 + 256*1024*1*1 + ...
        1024*256*1*1 + 256*256*3*3 + 256*1024*1*1 + 1024*256*1*1 + 256*256*3*3 + 256*1024*1*1 + ...
        1024*256*1*1 + 256*256*3*3 + 256*1024*1*1 + 1024*256*1*1 + 256*256*3*3 + 256*1024*1*1 + ...
        1024*512*1*1 + 512*512*3*3 + 512*2048*1*1 + 1024*2048*1*1 + 2048*512*1*1 + 512*512*3*3 + 512*2048*1*1 + ...
        2048*512*1*1 + 512*512*3*3 + 512*2048*1*1 + ...
        2048*1000)/10^6;

    p_cur = (3*64*7*7 + ...
        64*(64-l(1))*1*1 + (64-l(1))*(64-l(2))*3*3 + (64-l(2))*(256-l(3))*1*1 + 64*(256-l(9))*1*1 + ...
        (256-l(3))*(64-l(4))*1*1 + (64-l(4))*(64-l(5))*3*3 + (64-l(5))*(256-l(6))*1*1 + ...
        (256-l(6))*(64-l(7))*1*1 + (64-l(7))*(64-l(8))*3*3 + (64-l(8))*(256-l(9))*1*1 + ...
        (256-l(9))*(128-l(10))*1*1 + (128-l(10))*(128-l(11))*3*3 + (128-l(11))*(512-l(12))*1*1 + (256-l(9))*(512-l(21))*1*1 + ...
        (512-l(12))*(128-l(13))*1*1 + (128-l(13))*(128-l(14))*3*3 + (128-l(14))*(512-l(15))*1*1 + ...
        (512-l(15))*(128-l(16))*1*1 + (128-l(16))*(128-l(17))*3*3 + (128-l(17))*(512-l(18))*1*1 + ...
        (512-l(18))*(128-l(19))*1*1 + (128-l(19))*(128-l(20))*3*3 + (128-l(20))*(512-l(21))*1*1 + ...
        (512-l(21))*(256-l(22))*1*1 + (256-l(22))*(256-l(23))*3*3 + (256-l(23))*(1024-l(24))*1*1 + (512-l(21))*(1024-l(39))*1*1 + ...
        (1024-l(24))*(256-l(25))*1*1 + (256-l(25))*(256-l(26))*3*3 + (256-l(26))*(1024-l(27))*1*1 + ...
        (1024-l(27))*(256-l(28))*1*1 + (256-l(28))*(256-l(29))*3*3 + (256-l(29))*(1024-l(30))*1*1 + ...
        (1024-l(30))*(256-l(31))*1*1 + (256-l(31))*(256-l(32))*3*3 + (256-l(32))*(1024-l(33))*1*1 + ...
        (1024-l(33))*(256-l(34))*1*1 + (256-l(34))*(256-l(35))*3*3 + (256-l(35))*(1024-l(36))*1*1 + ...
        (1024-l(36))*(256-l(37))*1*1 + (256-l(37))*(256-l(38))*3*3 + (256-l(38))*(1024-l(39))*1*1 + ...
        (1024-l(39))*(512-l(40))*1*1 + (512-l(40))*(512-l(41))*3*3 + (512-l(41))*(2048-l(42))*1*1 + (1024-l(39))*(2048-l(48))*1*1 + ....
        (2048-l(42))*(512-l(43))*1*1 + (512-l(43))*(512-l(44))*3*3 + (512-l(44))*(2048-l(45))*1*1 + ...
        (2048-l(45))*(512-l(46))*1*1 + (512-l(46))*(512-l(47))*3*3 + (512-l(47))*(2048-l(48))*1*1 + ...
        (2048-l(48))*1000)/10^6;

    p_per = (p_ori - p_cur)/p_ori;


    f_ori = (3*64*7*7*112*112 + ...
        (64*64*1*1 + 64*64*3*3 + 64*256*1*1 + 64*256*1*1 + 256*64*1*1 + 64*64*3*3 + 64*256*1*1 + 256*64*1*1 + 64*64*3*3 + 64*256*1*1 + ...
        256*128*1*1)*56*56 + (128*128*3*3 + 128*512*1*1 + 256*512*1*1 + 512*128*1*1 + 128*128*3*3 + 128*512*1*1 + ...
        512*128*1*1 + 128*128*3*3 + 128*512*1*1 + 512*128*1*1 + 128*128*3*3 + 128*512*1*1 + ...
        512*256*1*1)*28*28 + (256*256*3*3 + 256*1024*1*1 + 512*1024*1*1 + 1024*256*1*1 + 256*256*3*3 + 256*1024*1*1 + ...
        1024*256*1*1 + 256*256*3*3 + 256*1024*1*1 + 1024*256*1*1 + 256*256*3*3 + 256*1024*1*1 + ...
        1024*256*1*1 + 256*256*3*3 + 256*1024*1*1 + 1024*256*1*1 + 256*256*3*3 + 256*1024*1*1 + ...
        1024*512*1*1)*14*14 + (512*512*3*3 + 512*2048*1*1 + 1024*2048*1*1 + 2048*512*1*1 + 512*512*3*3 + 512*2048*1*1 + ...
        2048*512*1*1 + 512*512*3*3 + 512*2048*1*1)*7*7 + ...
        2048*1000)/10^9;

    f_cur = (3*64*7*7*112*112 + ...
        (64*(64-l(1))*1*1 + (64-l(1))*(64-l(2))*3*3 + (64-l(2))*(256-l(3))*1*1 + 64*(256-l(9))*1*1 + ...
        (256-l(3))*(64-l(4))*1*1 + (64-l(4))*(64-l(5))*3*3 + (64-l(5))*(256-l(6))*1*1 + ...
        (256-l(6))*(64-l(7))*1*1 + (64-l(7))*(64-l(8))*3*3 + (64-l(8))*(256-l(9))*1*1 + ...
        (256-l(9))*(128-l(10))*1*1)*56*56 + ((128-l(10))*(128-l(11))*3*3 + (128-l(11))*(512-l(12))*1*1 + (256-l(9))*(512-l(21))*1*1 + ...
        (512-l(12))*(128-l(13))*1*1 + (128-l(13))*(128-l(14))*3*3 + (128-l(14))*(512-l(15))*1*1 + ...
        (512-l(15))*(128-l(16))*1*1 + (128-l(16))*(128-l(17))*3*3 + (128-l(17))*(512-l(18))*1*1 + ...
        (512-l(18))*(128-l(19))*1*1 + (128-l(19))*(128-l(20))*3*3 + (128-l(20))*(512-l(21))*1*1 + ...
        (512-l(21))*(256-l(22))*1*1)*28*28 + ((256-l(22))*(256-l(23))*3*3 + (256-l(23))*(1024-l(24))*1*1 + (512-l(21))*(1024-l(39))*1*1 + ...
        (1024-l(24))*(256-l(25))*1*1 + (256-l(25))*(256-l(26))*3*3 + (256-l(26))*(1024-l(27))*1*1 + ...
        (1024-l(27))*(256-l(28))*1*1 + (256-l(28))*(256-l(29))*3*3 + (256-l(29))*(1024-l(30))*1*1 + ...
        (1024-l(30))*(256-l(31))*1*1 + (256-l(31))*(256-l(32))*3*3 + (256-l(32))*(1024-l(33))*1*1 + ...
        (1024-l(33))*(256-l(34))*1*1 + (256-l(34))*(256-l(35))*3*3 + (256-l(35))*(1024-l(36))*1*1 + ...
        (1024-l(36))*(256-l(37))*1*1 + (256-l(37))*(256-l(38))*3*3 + (256-l(38))*(1024-l(39))*1*1 + ...
        (1024-l(39))*(512-l(40))*1*1)*14*14 + ((512-l(40))*(512-l(41))*3*3 + (512-l(41))*(2048-l(42))*1*1 + (1024-l(39))*(2048-l(48))*1*1 + ....
        (2048-l(42))*(512-l(43))*1*1 + (512-l(43))*(512-l(44))*3*3 + (512-l(44))*(2048-l(45))*1*1 + ...
        (2048-l(45))*(512-l(46))*1*1 + (512-l(46))*(512-l(47))*3*3 + (512-l(47))*(2048-l(48))*1*1)*7*7 + ...
        (2048-l(48))*1000)/10^9;

    f_per = (f_ori - f_cur)/f_ori;

    fprintf('ResNet-50: \n Parameter --- Ori: %.2fB, Cur: %.2fB, Rate: %.3f; \n FLOPs --- Ori: %.2fB, Cur: %.2fB, Rate: %.3f; \n', p_ori, p_cur, p_per, f_ori, f_cur, f_per);

end