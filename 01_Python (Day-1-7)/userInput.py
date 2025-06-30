p1Name = input("What is your name : ")
p1Age  = int(input("What is you age : "))

print(f" {p1Name} your age is {p1Age}")


p2Name = input("What is your name : ")
p2Age  = int(input("What is you age : "))

print(f"{p2Name} your age is {p2Age}")

if p1Age > p2Age:
    print(f"{p1Name} is older than {p2Name}")
    

elif  p1Age == p2Age:
    print(f"{p1Name} and {p2Name} are the same age")                    

else:
    print(f"{p2Name} is older than {p1Name}")