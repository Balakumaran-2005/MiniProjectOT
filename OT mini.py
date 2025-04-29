import numpy as np
import ot
import warnings

class Booking:
    def __init__(self, booking_id, customer_id, pick_loc, drop_loc, pick_time, drop_time, amount):
        self.booking_id = booking_id
        self.customer_id = customer_id
        self.pick_loc = pick_loc
        self.drop_loc = drop_loc
        self.pick_time = pick_time
        self.drop_time = drop_time
        self.amount = amount

class Taxi:
    def __init__(self, taxi_id):
        self.id = taxi_id
        self.location = 'A'
        self.total_earning = 0
        self.available_at = 0
        self.bookings = []

    def is_available(self, pickup_time):
        return self.available_at <= pickup_time

    def calculate_earnings(self, pick, drop):
        distance = abs(ord(pick) - ord(drop)) * 15
        return 100 + max(0, (distance - 5) * 10)

    def add_booking(self, booking):
        self.bookings.append(booking)
        self.total_earning += booking.amount
        self.location = booking.drop_loc
        self.available_at = booking.drop_time

def create_cost_matrix(taxis, requests):
    cost_matrix = []
    for taxi in taxis:
        row = []
        for req in requests:
            if taxi.is_available(req['pickup_time']):
                cost = abs(ord(taxi.location) - ord(req['pickup']))
            else:
                cost = 1e6  # Assign a large cost for unavailable taxis
            row.append(cost)
        cost_matrix.append(row)
    return np.array(cost_matrix)



def assign_taxis_ot(taxis, requests):
    cost_matrix = create_cost_matrix(taxis, requests)
    
    # Ensure the cost matrix is of type float64 to allow float operations
    cost_matrix = cost_matrix.astype(np.float64)  # Convert to float64 type
    
    # Add a small constant to avoid division by zero issues
    cost_matrix += 1e-6  # Stabilize the cost matrix
    
    # Ensure no NaN or inf values are present
    if np.any(np.isnan(cost_matrix)) or np.any(np.isinf(cost_matrix)):
        print("Error: Cost matrix contains NaN or infinity values.")
        return

    # Supply and demand distributions
    n, m = cost_matrix.shape
    supply = np.ones(n) / n  # Equal supply for taxis
    demand = np.ones(m) / m  # Equal demand for requests
    
    # Adjust regularization to improve stability
    reg = 0.1  # You can try different values, e.g., 0.1 or 0.5
    
    try:
        # Suppress warnings temporarily during Sinkhorn computation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transport_plan = ot.sinkhorn(supply, demand, cost_matrix, reg=reg)
    except Exception as e:
        print(f"Error during optimal transport calculation: {e}")
        return
    
    # If transport plan calculation succeeds, process the assignments
    try:
        assignments = transport_plan.argmax(axis=0)
    except Exception as e:
        print(f"Error during assignment extraction: {e}")
        return

    for j, i in enumerate(assignments):
        taxi = taxis[i]
        req = requests[j]
        if taxi.is_available(req['pickup_time']):
            drop_time = req['pickup_time'] + abs(ord(req['pickup']) - ord(req['drop']))
            amount = taxi.calculate_earnings(req['pickup'], req['drop'])
            booking_id = len(taxi.bookings) + 1
            booking = Booking(booking_id, req['customer_id'], req['pickup'], req['drop'], req['pickup_time'], drop_time, amount)
            taxi.add_booking(booking)
            print(f"\nTaxi-{taxi.id} is allocated to Customer-{req['customer_id']}.\n")
        else:
            print(f"\nNo available taxi for Customer-{req['customer_id']}.\n")


def display_taxi_details(taxis):
    for taxi in taxis:
        print(f"\nTaxi-{taxi.id} | Total Earnings: Rs.{taxi.total_earning}")
        print(f"{'BookingID':<10}{'CustomerID':<12}{'From':<6}{'To':<6}{'PickupTime':<12}{'DropTime':<10}{'Amount'}")
        for b in taxi.bookings:
            print(f"{b.booking_id:<10}{b.customer_id:<12}{b.pick_loc:<6}{b.drop_loc:<6}{b.pick_time:<12}{b.drop_time:<10}{b.amount}")

# MAIN PROGRAM
def main():
    customer_id = 0
    taxis = []
    requests = []

    num_taxis = int(input("Enter number of taxis: "))
    for i in range(num_taxis):
        taxis.append(Taxi(i + 1))

    while True:
        print("\n1. Book Taxi\n2. Display Taxi Details\n3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            customer_id += 1
            pickup = input("Enter Pickup Point (A-F): ").strip().upper()
            drop = input("Enter Drop Point (A-F): ").strip().upper()
            pickup_time = int(input("Enter Pickup Time (0-23): "))

            request = {
                'customer_id': customer_id,
                'pickup': pickup,
                'drop': drop,
                'pickup_time': pickup_time
            }

            assign_taxis_ot(taxis, [request])  # Process one request at a time

        elif choice == '2':
            display_taxi_details(taxis)

        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid option!")

if __name__ == "__main__":
    # Suppress warnings to keep the console output clean
    warnings.filterwarnings("ignore", category=UserWarning, message=".*numerical errors.*")
    main()
