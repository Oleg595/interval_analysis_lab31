%correction right part

midA = [2 2;
    1 -2;
    1 0]

b = [fixed.Interval(1.289, 4.711);
    fixed.Interval(-0.711, 0.711);
    fixed.Interval(0.289, 4.711)]

argmax = [1.06;
    0.44]

tolmax = 0.1
con = cond(midA)
b_norm = norm2(b)
argmax_norm = norm(argmax)

rve = con * tolmax
ive = (con * argmax_norm * tolmax / b_norm)

down_line = argmax(1) - ive;
high_line = argmax(1) + ive;
left_line = argmax(2) - ive;
right_line = argmax(2) + ive;
figure;
line1 = line([left_line down_line],[left_line high_line]);
line2 = line([left_line high_line],[right_line high_line]);
line3 = line([right_line high_line],[right_line down_line]);
line4 = line([right_line down_line],[left_line down_line]);

%correction matrix

midA = [2 2;
    1 -2;
    1 0]

b = [fixed.Interval(2, 4);
    fixed.Interval(0, 0);
    fixed.Interval(1, 4)]

argmax = [1;
    0.5]

tolmax = 5.821 .* 10 .^ (-11)
con = cond(midA)
b_norm = norm2(b)
argmax_norm = norm(argmax)

rve = con * tolmax
ive = (con * argmax_norm * tolmax / b_norm)

function n = norm2(vector)
    n = 0
    for i = 1:numel(vector)
        if abs(vector(i).LeftEnd) > abs(vector(i).RightEnd)
            n = n + sqrt(abs(vector(i).LeftEnd) .^ 2)
       
        else
            n = n + sqrt(abs(vector(i).RightEnd))
        end
    end
    n = sqrt(n)
end
