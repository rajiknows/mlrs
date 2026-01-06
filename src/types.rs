// 32-bit boolean type
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct B32(pub u32);

impl B32 {
    pub const FALSE: Self = Self(0);
    pub const TRUE: Self = Self(1);

    pub fn new(v: bool) -> Self {
        Self(v as u32)
    }

    pub fn get(&self) -> bool {
        self.0 != 0
    }
}

impl From<bool> for B32 {
    fn from(value: bool) -> Self {
        Self::new(value)
    }
}

impl From<B32> for bool {
    fn from(value: B32) -> Self {
        value.get()
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct B8(pub u8);

impl B8 {
    pub const FALSE: Self = Self(0);
    pub const TRUE: Self = Self(1);

    pub fn new(v: bool) -> Self {
        Self(v as u8)
    }

    pub fn get(&self) -> bool {
        self.0 != 0
    }
}

impl From<bool> for B8 {
    fn from(value: bool) -> Self {
        Self::new(value)
    }
}

impl From<B8> for bool {
    fn from(value: B8) -> Self {
        value.get()
    }
}
