import {
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib";
import React, { ReactNode } from "react";

interface State {
  isFocused: boolean;
}

class AudioPlayer extends StreamlitComponentBase<State> {
  public state = { isFocused: false };

  public render = (): ReactNode => {
    const audioUrls: string[] = this.props.args["audioUrls"];
    const segmentNames: string[] = this.props.args["segmentNames"] || [];
    const { theme } = this.props;

    // Define a basic style for the buttons
    const style: React.CSSProperties = {
      border: '1px solid gray',
      padding: '8px 16px',
      margin: '4px',
      cursor: 'pointer',
      backgroundColor: theme ? theme.backgroundColor : '#f0f0f0',
      color: theme ? theme.textColor : '#333',
    };

    return (
      <div>
        {audioUrls.map((url: string, index: number) => (
          <button
            key={index}
            style={style}
            onClick={() => this.playAudio(index)}
            disabled={this.props.disabled}
            onFocus={this._onFocus}
            onBlur={this._onBlur}
          >
            Play {segmentNames[index] || `Point ${index}`} {/* Display segment name or fallback to index */}
          </button>
        ))}
      </div>
    );
  }

  /** Play audio for a specific point */
  private playAudio = (index: number): void => {
    const audioUrl = this.props.args.audioUrls[index];
    const audio = new Audio(audioUrl);
    audio.play();
  };

  /** Focus handler for our button. */
  private _onFocus = (): void => {
    this.setState({ isFocused: true });
  };

  /** Blur handler for our button. */
  private _onBlur = (): void => {
    this.setState({ isFocused: false });
  };
}

export default withStreamlitConnection(AudioPlayer);
