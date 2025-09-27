import React from 'react';
import { X } from 'lucide-react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
}

const Modal: React.FC<ModalProps> = ({ isOpen, onClose, children }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal Content */}
      <div className="relative z-10 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="bg-white rounded-2xl shadow-2xl">
          {/* Close Button */}
          <button
            onClick={onClose}
            className="absolute top-4 right-4 z-20 p-2 rounded-full bg-slate-100 hover:bg-slate-200 transition-colors"
          >
            <X className="w-5 h-5 text-slate-600" />
          </button>
          
          {children}
        </div>
      </div>
    </div>
  );
};

export default Modal;