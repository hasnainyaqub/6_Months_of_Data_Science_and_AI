#Dictionary

total_bill = 0
print("Welcome to our  restaurant!")

print("Here are our menu items and prices:")
print("1.Samosa: 25\n2.Pakor: 50\n3.Burge: 375\n4.Pizz: 750\n5.Frie: 150")

item_1 =  input("Please enter the name of the item you would like to order: ")



menu = {
    '1': 25,
    '2': 50,
    '3': 375,
    '4': 750,
    '5': 150
 }      

if item_1 in menu:
    total_bill += menu[item_1]
else:
    print("Not availeble")

print(total_bill)