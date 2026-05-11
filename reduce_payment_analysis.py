"""
Calculate prepayment needed to reduce monthly payment to $10k
"""

def calculate_monthly_payment(principal, annual_rate, years):
    """Calculate monthly payment for a loan"""
    monthly_rate = annual_rate / 100 / 12
    num_payments = years * 12

    if monthly_rate == 0:
        return principal / num_payments

    payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / \
              ((1 + monthly_rate)**num_payments - 1)
    return payment

def calculate_prepayment_needed(current_balance, annual_rate, target_payment, loan_years):
    """Calculate how much to prepay to achieve target monthly payment"""
    # Binary search for the right balance
    low, high = 0, current_balance
    tolerance = 100  # $100 tolerance

    while high - low > tolerance:
        mid = (low + high) / 2
        new_balance = current_balance - mid
        payment = calculate_monthly_payment(new_balance, annual_rate, loan_years)

        if payment > target_payment:
            low = mid
        else:
            high = mid

    prepayment = (low + high) / 2
    new_balance = current_balance - prepayment
    actual_payment = calculate_monthly_payment(new_balance, annual_rate, loan_years)

    return prepayment, new_balance, actual_payment

def analyze_payment_reduction(current_balance, annual_rate, current_payment, target_payment):
    """Analyze different scenarios to reduce monthly payment"""

    print("=" * 80)
    print("MONTHLY PAYMENT REDUCTION ANALYSIS")
    print("=" * 80)
    print(f"\nCurrent Loan Balance: ${current_balance:,.2f}")
    print(f"Current Monthly Payment: ${current_payment:,.2f}")
    print(f"Target Monthly Payment: ${target_payment:,.2f}")
    print(f"Payment Reduction Goal: ${current_payment - target_payment:,.2f}/month")
    print(f"Interest Rate: {annual_rate}%")

    print("\n" + "=" * 80)
    print("SCENARIOS: How much to prepay to achieve $10k/month payment")
    print("=" * 80)

    scenarios = [
        ("30-year re-amortization", 30),
        ("25-year re-amortization", 25),
        ("20-year re-amortization", 20),
        ("15-year re-amortization", 15),
        ("10-year re-amortization", 10),
    ]

    results = []

    for scenario_name, years in scenarios:
        prepayment, new_balance, actual_payment = calculate_prepayment_needed(
            current_balance, annual_rate, target_payment, years
        )

        # Calculate total cost
        total_paid = prepayment + (actual_payment * years * 12)
        total_interest = total_paid - current_balance

        results.append({
            'name': scenario_name,
            'years': years,
            'prepayment': prepayment,
            'new_balance': new_balance,
            'payment': actual_payment,
            'total_paid': total_paid,
            'total_interest': total_interest
        })

        print(f"\n{scenario_name.upper()}")
        print("-" * 80)
        print(f"  Prepayment needed: ${prepayment:,.2f}")
        print(f"  New loan balance: ${new_balance:,.2f}")
        print(f"  New monthly payment: ${actual_payment:,.2f}")
        print(f"  Loan term: {years} years ({years * 12} months)")
        print(f"  Total you'll pay: ${total_paid:,.2f}")
        print(f"    (${prepayment:,.2f} upfront + ${actual_payment * years * 12:,.2f} in payments)")
        print(f"  Total interest over life: ${total_interest:,.2f}")

    # Find optimal scenario
    print("\n" + "=" * 80)
    print("COMPARISON & RECOMMENDATIONS")
    print("=" * 80)

    # Sort by prepayment needed
    sorted_by_prepay = sorted(results, key=lambda x: x['prepayment'])

    print(f"\nLeast prepayment needed:")
    best = sorted_by_prepay[0]
    print(f"  → {best['name']}: ${best['prepayment']:,.2f}")
    print(f"    But you'll pay ${best['total_interest']:,.2f} in total interest")

    # Sort by total interest
    sorted_by_interest = sorted(results, key=lambda x: x['total_interest'])

    print(f"\nLeast total interest paid:")
    best = sorted_by_interest[0]
    print(f"  → {best['name']}: ${best['total_interest']:,.2f} interest")
    print(f"    Requires ${best['prepayment']:,.2f} prepayment")

    # Middle ground
    print(f"\nBalanced option (20-year):")
    balanced = [r for r in results if r['years'] == 20][0]
    print(f"  Prepayment: ${balanced['prepayment']:,.2f}")
    print(f"  Monthly payment: ${balanced['payment']:,.2f}")
    print(f"  Total interest: ${balanced['total_interest']:,.2f}")
    print(f"  Payoff time: {balanced['years']} years")

    print("\n" + "=" * 80)
    print("ALTERNATIVE: KEEP PAYMENT AT $14K")
    print("=" * 80)

    # Calculate what happens if they just keep paying $14k on current balance
    monthly_rate = annual_rate / 100 / 12
    balance = current_balance
    months = 0
    total_interest = 0

    while balance > 0 and months < 360:
        interest = balance * monthly_rate
        principal = current_payment - interest
        total_interest += interest
        balance -= principal
        months += 1

    print(f"\nIf you keep paying ${current_payment:,.2f}/month on ${current_balance:,.2f}:")
    print(f"  Loan paid off in: {months} months ({months/12:.1f} years)")
    print(f"  Total interest: ${total_interest:,.2f}")
    print(f"  Total paid: ${current_balance + total_interest:,.2f}")
    print(f"\n  → This saves you ${sorted_by_interest[0]['total_interest'] - total_interest:,.2f}")
    print(f"     compared to the best prepayment scenario!")
    print(f"  → But requires keeping payment at ${current_payment:,.2f} (not reducing to ${target_payment:,.2f})")

    print("\n" + "=" * 80)
    print("CASH FLOW ANALYSIS")
    print("=" * 80)

    monthly_savings = current_payment - target_payment
    print(f"\nBy reducing payment to ${target_payment:,.2f}, you'll save:")
    print(f"  ${monthly_savings:,.2f}/month = ${monthly_savings * 12:,.2f}/year")
    print(f"\nOver 10 years, that's ${monthly_savings * 120:,.2f} in freed-up cash flow")

    print("\n" + "=" * 80)
    print("KEY QUESTIONS TO CONSIDER")
    print("=" * 80)
    print(f"\n1. Do you NEED to reduce your payment, or do you WANT to?")
    print(f"   - If you can afford ${current_payment:,.2f}, keeping it saves the most money")
    print(f"   - If you need cash flow relief, prepayment + re-amortization makes sense")
    print(f"\n2. What will you do with the ${monthly_savings:,.2f}/month savings?")
    print(f"   - Invest it? (Could earn 7-10% returns)")
    print(f"   - Spend it? (Lifestyle improvement)")
    print(f"   - Save it? (Emergency fund, future goals)")
    print(f"\n3. Do you have the cash available for prepayment?")
    print(f"   - Least needed: ${sorted_by_prepay[0]['prepayment']:,.2f} ({sorted_by_prepay[0]['name']})")
    print(f"   - Most efficient: ${sorted_by_interest[0]['prepayment']:,.2f} ({sorted_by_interest[0]['name']})")
    print(f"\n4. Will you actually re-amortize or refinance?")
    print(f"   - Refinancing costs: $5k-$15k in closing costs")
    print(f"   - Loan modification: May be cheaper if lender allows")
    print(f"   - DIY prepayment (no re-amortization): Payment stays at ${current_payment:,.2f}")

if __name__ == "__main__":
    CURRENT_BALANCE = 2_550_000
    ANNUAL_RATE = 5.25
    CURRENT_PAYMENT = 14_000
    TARGET_PAYMENT = 10_000

    analyze_payment_reduction(CURRENT_BALANCE, ANNUAL_RATE, CURRENT_PAYMENT, TARGET_PAYMENT)
