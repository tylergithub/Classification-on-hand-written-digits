
x = -10:1:100;
y = test(x);
plot(x,y);





function y = test(x)

for i = 1:size(x,2)
    if x(i) < 50
        y(i) = 0;
    else
        y(i) = x(i);
    end
end
end