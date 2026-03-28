// This component allows users to book a cleaning service by selecting a service type, specifying the number of rooms and bathrooms, and choosing any additional add-ons. The total price is calculated based on the selected options and displayed in a summary card.

import React, { useState, useMemo } from 'react';
import seedData from '../../../shared/seedData.json';
import { calculateTotal } from '../../../shared/pricingLogic';

const BookService = () => {
  const [selectedService, setSelectedService] = useState(seedData.services[0]);
  const [rooms, setRooms] = useState(1);
  const [baths, setBaths] = useState(1);
  const [selectedAddons, setSelectedAddons] = useState([]);

  // Memoize the total so it only recalculates when inputs change
  const total = useMemo(() => 
    calculateTotal(selectedService, rooms, baths, selectedAddons),
    [selectedService, rooms, baths, selectedAddons]
  );

  const toggleAddon = (addon) => {
    setSelectedAddons(prev => 
      prev.find(a => a.id === addon.id) 
        ? prev.filter(a => a.id !== addon.id) 
        : [...prev, addon]
    );
  };

  return (
    <div className="max-w-4xl mx-auto p-8 flex flex-col md:flex-row gap-8">
      {/* Configuration Section */}
      <div className="flex-1 space-y-6">
        <h1 className="text-3xl font-bold text-gray-900">Book a Cleaning</h1>
        
        <div>
          <label className="block text-sm font-medium text-gray-700">Select Service</label>
          <select 
            className="mt-1 block w-full p-2 border rounded-md"
            onChange={(e) => setSelectedService(seedData.services.find(s => s.id === e.target.value))}
          >
            {seedData.services.map(s => <option key={s.id} value={s.id}>{s.title}</option>)}
          </select>
        </div>

        <div className="flex gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700">Bedrooms</label>
            <input type="number" min="1" value={rooms} onChange={(e) => setRooms(parseInt(e.target.value))} className="mt-1 w-full p-2 border rounded-md" />
          </div>
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700">Bathrooms</label>
            <input type="number" min="1" value={baths} onChange={(e) => setBaths(parseInt(e.target.value))} className="mt-1 w-full p-2 border rounded-md" />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Available Add-ons</label>
          <div className="grid grid-cols-2 gap-2">
            {selectedService.addons.map(addon => (
              <button 
                key={addon.id}
                onClick={() => toggleAddon(addon)}
                className={`p-3 text-sm border rounded-lg text-left ${selectedAddons.find(a => a.id === addon.id) ? 'bg-blue-50 border-blue-500' : 'bg-white'}`}
              >
                {addon.name} (+${addon.price})
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Pricing Summary Card */}
      <div className="w-full md:w-80 bg-gray-50 p-6 rounded-xl border border-gray-200 h-fit">
        <h2 className="text-xl font-semibold mb-4">Summary</h2>
        <div className="space-y-2 border-b pb-4 mb-4 text-sm text-gray-600">
          <div className="flex justify-between"><span>Base Price</span><span>${selectedService.basePrice}</span></div>
          <div className="flex justify-between"><span>Rooms/Baths</span><span>+${(rooms * selectedService.modifiers.pricePerBedroom) + (baths * selectedService.modifiers.pricePerBathroom)}</span></div>
          {selectedAddons.map(a => (
            <div key={a.id} className="flex justify-between"><span>{a.name}</span><span>+${a.price}</span></div>
          ))}
        </div>
        <div className="flex justify-between items-center mb-6">
          <span className="text-lg font-bold text-gray-900">Total</span>
          <span className="text-2xl font-bold text-blue-600">${total}</span>
        </div>
        <button className="w-full bg-blue-600 text-white py-3 rounded-lg font-bold hover:bg-blue-700 transition">
          Confirm & Pay
        </button>
      </div>
    </div>
  );
};

export default BookService;