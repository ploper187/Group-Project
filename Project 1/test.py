l = list(range(10))
for i in range(len(l)):
	if l[i] % 2 == 0:
		print("removing", l[i])
		l.remove(l[i])
		i -= 2
	else:
		print(l[i])