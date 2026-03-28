// components/CheckoutForm.jsx

import { CardElement, useStripe, useElements } from '@stripe/react-stripe-js';

const CheckoutForm = ({ totalAmount }) => {
  const stripe = useStripe();
  const elements = useElements();

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!stripe || !elements) return;

    // 1. Request a Client Secret from your backend
    const { clientSecret } = await fetch('/api/create-payment-intent', {
      method: 'POST',
      body: JSON.stringify({ amount: totalAmount }),
    }).then(res => res.json());

    // 2. Confirm the payment with Stripe
    const result = await stripe.confirmCardPayment(clientSecret, {
      payment_method: {
        card: elements.getElement(CardElement),
      }
    });

    if (result.error) {
      console.log(result.error.message);
    } else {
      if (result.paymentIntent.status === 'succeeded') {
        alert("Booking Secured! Funds held until service completion.");
      }
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="p-3 border rounded-md bg-white">
        <CardElement options={{ style: { base: { fontSize: '16px' } } }} />
      </div>
      <button className="w-full bg-green-600 text-white py-3 rounded-lg font-bold">
        Secure Booking (${totalAmount})
      </button>
    </form>
  );
};

export default CheckoutForm;
