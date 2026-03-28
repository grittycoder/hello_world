// shared/pricingLogic.js

export const calculateTotal = (service, rooms, baths, addons = []) => {
  if (!service) return 0;
  
  const base = service.basePrice || 0;
  const roomCost = (rooms || 0) * (service.modifiers?.pricePerBedroom || 0);
  const bathCost = (baths || 0) * (service.modifiers?.pricePerBathroom || 0);
  const addonCost = addons.reduce((sum, addon) => sum + (addon.price || 0), 0);
  
  return base + roomCost + bathCost + addonCost;
};

export const calculateDiscount = (total, discount) => {
  if (!discount) return 0;
  if (discount.type === 'percentage') {
    return total * (discount.value / 100);
  } else if (discount.type === 'fixed') {
    return discount.value;
  }
  return 0;
};

export const calculateFinalPrice = (service, rooms, baths, addons = [], discount) => {
  const total = calculateTotal(service, rooms, baths, addons);
  const discountAmount = calculateDiscount(total, discount);
  return Math.max(total - discountAmount, 0); // Ensure final price is not negative
}; 