//! Concrete test scenes (the analog of the per-project TestScene
//! implementations in osu). Registering a new scene: implement `TestCase`
//! here, re-export it below and register it in `Game`.

mod key_combination;
mod sprite_batch;

pub use key_combination::KeyCombinationScreen;
pub use sprite_batch::SpriteBatchScreen;
