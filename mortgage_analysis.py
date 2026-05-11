"""
Mortgage Payoff Analysis
Compare interest savings from paying $200k off principal
"""

def calculate_remaining_balance(principal, monthly_rate, monthly_payment, months):
    """Calculate remaining balance after given months

    Args:
        principal: Starting loan balance
        monthly_rate: Monthly interest rate (already divided by 12)
        monthly_payment: Monthly payment amount
        months: Number of months to calculate
    """
    balance = principal
    total_interest = 0

    for month in range(months):
        if balance <= 0:
            break
        interest_payment = balance * monthly_rate
        principal_payment = monthly_payment - interest_payment
        total_interest += interest_payment
        balance -= principal_payment

    return balance, total_interest

def analyze_scenarios(current_balance, annual_rate, monthly_payment, prepayment, horizon_years):
    """Analyze current vs prepayment scenarios"""
    monthly_rate = annual_rate / 100 / 12
    horizon_months = horizon_years * 12

    print("=" * 70)
    print("MORTGAGE PAYOFF ANALYSIS")
    print("=" * 70)
    print(f"\nCurrent Loan Balance: ${current_balance:,.2f}")
    print(f"Annual Interest Rate: {annual_rate}%")
    print(f"Monthly Payment: ${monthly_payment:,.2f}")
    print(f"Analysis Horizon: {horizon_years} years ({horizon_months} months)")
    print(f"Proposed Principal Payoff: ${prepayment:,.2f}")

    # Scenario 1: Current loan
    print("\n" + "=" * 70)
    print("SCENARIO 1: Continue with current loan")
    print("=" * 70)
    balance1, interest1 = calculate_remaining_balance(
        current_balance, monthly_rate, monthly_payment, horizon_months
    )
    principal_paid1 = current_balance - balance1
    total_paid1 = monthly_payment * horizon_months

    print(f"Total payments over {horizon_years} years: ${total_paid1:,.2f}")
    print(f"Total interest paid: ${interest1:,.2f}")
    print(f"Principal paid down: ${principal_paid1:,.2f}")
    print(f"Remaining balance after {horizon_years} years: ${balance1:,.2f}")

    # Scenario 2: Pay off $200k now
    print("\n" + "=" * 70)
    print(f"SCENARIO 2: Pay ${prepayment:,.2f} off principal now")
    print("=" * 70)
    new_balance = current_balance - prepayment
    print(f"New loan balance: ${new_balance:,.2f}")

    balance2, interest2 = calculate_remaining_balance(
        new_balance, monthly_rate, monthly_payment, horizon_months
    )
    principal_paid2 = new_balance - balance2
    total_paid2 = monthly_payment * horizon_months + prepayment

    print(f"Total payments over {horizon_years} years: ${total_paid2:,.2f}")
    print(f"  (includes ${prepayment:,.2f} upfront payment)")
    print(f"Total interest paid: ${interest2:,.2f}")
    print(f"Principal paid down: ${principal_paid2 + prepayment:,.2f}")
    print(f"Remaining balance after {horizon_years} years: ${balance2:,.2f}")

    # Comparison
    print("\n" + "=" * 70)
    print("SAVINGS ANALYSIS")
    print("=" * 70)
    interest_savings = interest1 - interest2
    remaining_balance_reduction = balance1 - balance2

    print(f"Interest saved over {horizon_years} years: ${interest_savings:,.2f}")
    print(f"Additional principal paid down: ${remaining_balance_reduction:,.2f}")
    print(f"Total benefit (interest + equity): ${interest_savings + remaining_balance_reduction:,.2f}")

    # ROI Analysis
    print("\n" + "=" * 70)
    print("RETURN ON INVESTMENT")
    print("=" * 70)
    roi_percentage = (interest_savings / prepayment) * 100
    annualized_roi = roi_percentage / horizon_years

    print(f"ROI on ${prepayment:,.2f} payment: {roi_percentage:.2f}% over {horizon_years} years")
    print(f"Annualized ROI: {annualized_roi:.2f}% per year")
    print(f"\nThis is effectively a guaranteed {annualized_roi:.2f}% annual return")

    # Opportunity Cost
    print("\n" + "=" * 70)
    print("OPPORTUNITY COST CONSIDERATIONS")
    print("=" * 70)
    print(f"Compare {annualized_roi:.2f}% guaranteed return vs alternatives:")
    print(f"  - Stock market (historical ~10% annually, but volatile)")
    print(f"  - High-yield savings (~4-5% currently)")
    print(f"  - Other investments")
    print(f"\nThe mortgage payoff provides a GUARANTEED {annualized_roi:.2f}% return")
    print(f"Tax considerations: Mortgage interest may be tax-deductible")

    # Payoff timeline
    print("\n" + "=" * 70)
    print("LOAN PAYOFF TIMELINE")
    print("=" * 70)

    # Calculate months to payoff for both scenarios
    def months_to_payoff(balance, rate, payment):
        months = 0
        while balance > 0 and months < 360:
            interest = balance * rate
            principal = payment - interest
            balance -= principal
            months += 1
        return months

    months1 = months_to_payoff(current_balance, monthly_rate, monthly_payment)
    months2 = months_to_payoff(new_balance, monthly_rate, monthly_payment)

    print(f"Current scenario: {months1} months ({months1/12:.1f} years) to payoff")
    print(f"With ${prepayment:,.2f} payoff: {months2} months ({months2/12:.1f} years) to payoff")
    print(f"Time saved: {months1 - months2} months ({(months1 - months2)/12:.1f} years)")

    # Full loan lifetime analysis
    print("\n" + "=" * 70)
    print("FULL LOAN LIFETIME ANALYSIS (Until Payoff)")
    print("=" * 70)

    # Calculate total interest over full loan life
    _, total_interest_full1 = calculate_remaining_balance(
        current_balance, monthly_rate, monthly_payment, months1
    )
    _, total_interest_full2 = calculate_remaining_balance(
        new_balance, monthly_rate, monthly_payment, months2
    )

    lifetime_interest_savings = total_interest_full1 - total_interest_full2
    lifetime_roi = (lifetime_interest_savings / prepayment) * 100

    print(f"Scenario 1 - Total interest over life of loan: ${total_interest_full1:,.2f}")
    print(f"Scenario 2 - Total interest over life of loan: ${total_interest_full2:,.2f}")
    print(f"\nLifetime interest savings: ${lifetime_interest_savings:,.2f}")
    print(f"Lifetime ROI: {lifetime_roi:.2f}% total")
    print(f"Time until payoff reduced by: {(months1-months2)/12:.1f} years")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"Over {horizon_years}-year horizon:")
    print(f"  Interest saved: ${interest_savings:,.2f}")
    print(f"  Annualized return: {annualized_roi:.2f}%")
    print(f"\nOver full loan lifetime:")
    print(f"  Interest saved: ${lifetime_interest_savings:,.2f}")
    print(f"  Total return: {lifetime_roi:.2f}%")
    print(f"\n{'⚠ CAUTION' if annualized_roi < 4 else '✓ GOOD OPTION'}:")
    print(f"  A {annualized_roi:.2f}% annual return is {'VERY LOW' if annualized_roi < 2 else 'modest'}.")
    print(f"  You can likely earn 4-5% risk-free in high-yield savings.")
    print(f"  Historical stock market returns average ~10% annually.")
    print(f"\nConsider:")
    print(f"  ✓ Keep the $200k invested if you can earn >0.5% annually")
    print(f"  ✓ High-yield savings (~4-5%) would earn ~$80k-$120k more")
    print(f"  ✓ Maintain liquidity for emergencies and opportunities")
    print(f"  ✗ Only prepay if you value debt reduction over returns")
    print(f"  ✗ Or if you have no better investment options")

if __name__ == "__main__":
    # Your mortgage details
    CURRENT_BALANCE = 2_550_000  # $2.55M
    ANNUAL_RATE = 5.25  # 5.25%
    MONTHLY_PAYMENT = 14_000  # $14,000
    PREPAYMENT = 200_000  # $200k
    HORIZON_YEARS = 10  # 10 year horizon

    analyze_scenarios(
        CURRENT_BALANCE,
        ANNUAL_RATE,
        MONTHLY_PAYMENT,
        PREPAYMENT,
        HORIZON_YEARS
    )
